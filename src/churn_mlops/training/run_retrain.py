"""Orquesta el reentreno rápido de Fase 6: reusa los últimos hiperparámetros
ganadores de Optuna (models/best_params.json, persistidos por
run_training.py::save_best_params) — SIN volver a correr la búsqueda completa
(ADR 0012). Refitea sobre todo train.parquet, evalúa una única vez sobre
test.parquet, guarda el pipeline y registra una versión nueva en el Model
Registry, igual que run_training.py pero saltando run_optuna_search.

Es el paso que reemplaza a `dvc repro train_model` cuando el reentreno lo
dispara un drift detectado (caso normal, ver monitoring/run_retrain_check.py);
`dvc repro train_model` queda reservado para cuando se quiere forzar un
re-tune completo con Optuna (caso excepcional). Entrypoint:

    uv run python -m churn_mlops.training.run_retrain
"""

import io
import json
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd

# La consola de Windows usa cp1252 por defecto, que no soporta los emojis que
# mlflow imprime al terminar un run (ej. 🏃). Sin esto, el script crashea después
# de loguear el run exitosamente. El isinstance narrowa sys.stdout a
# TextIOWrapper para mypy (TextIOBase no declara reconfigure).
if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.pipeline import Pipeline

from churn_mlops.config import load_yaml_config
from churn_mlops.training.dataset import load_processed_datasets
from churn_mlops.training.evaluation import compute_classification_metrics
from churn_mlops.training.run_training import (
    configure_mlflow,
    refit_best_pipeline,
    save_pipeline_artifact,
)


def load_best_params(path: str | Path, fixed_params: dict[str, Any]) -> dict[str, Any]:
    """Lee models/best_params.json (persistido por la última corrida de
    run_training.py) y hace merge con fixed_params, mismo orden que
    run_training.py::main() ({**fixed_params, **persisted}). Lanza
    FileNotFoundError con mensaje explícito si todavía no corrió ningún
    tuning completo."""
    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(
            f"No existe {params_path} — correr primero `dvc repro train_model` "
            "(o `uv run python -m churn_mlops.training.run_training`) al menos "
            "una vez para generar hiperparámetros ganadores antes de reusarlos."
        )
    persisted = json.loads(params_path.read_text(encoding="utf-8"))
    return {**fixed_params, **persisted}


def evaluate_and_log_refit_model(
    pipeline: Pipeline,
    best_params: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Evalúa una única vez sobre test.parquet, loguea params/métricas/modelo a
    MLflow y registra la versión en el Model Registry. Análogo a
    run_training.py::evaluate_and_log_final_model, pero sin cv_best_f1 (no hubo
    cross-validation acá) y con run_name distinto para diferenciar en MLflow
    los runs que vinieron de tuning completo ("lightgbm-tuned") de los que
    reusaron hiperparámetros ("lightgbm-reused-params")."""
    with mlflow.start_run(run_name="lightgbm-reused-params") as run:
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(
            y_test.to_numpy(), y_pred, y_proba
        )

        mlflow.log_params({f"lightgbm__{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(test_metrics)

        # serialization_format="pickle": el default (skops) rechaza tipos internos
        # de ColumnTransformer/OneHotEncoder (ej. numpy.dtype) como "no confiables".
        # Es nuestro propio modelo, no uno de terceros, así que pickle es adecuado.
        mlflow.sklearn.log_model(
            pipeline.named_steps["preprocess"],
            artifact_path="preprocessor",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        mlflow.lightgbm.log_model(
            pipeline.named_steps["model"],
            artifact_path="model",
            registered_model_name=training_config["registered_model_name"],
        )

        return {"run_id": run.info.run_id, **test_metrics}


def main() -> None:
    data_config = load_yaml_config("configs/data.yaml")
    training_config = load_yaml_config("configs/training.yaml")
    configure_mlflow(training_config)

    X_train, X_test, y_train, y_test = load_processed_datasets(data_config)
    high_cardinality_columns = training_config["high_cardinality_columns"]

    best_params = load_best_params(
        training_config["best_params_output_path"], training_config["fixed_params"]
    )

    pipeline = refit_best_pipeline(
        X_train, y_train, high_cardinality_columns, best_params
    )
    model_path = save_pipeline_artifact(pipeline, training_config["model_output_path"])

    result = evaluate_and_log_refit_model(
        pipeline, best_params, X_test, y_test, training_config
    )

    print(f"Hiperparámetros reusados: {best_params}")
    print(f"Métricas en test.parquet: {result}")
    print(f"Pipeline serializado en: {model_path}")


if __name__ == "__main__":
    main()
