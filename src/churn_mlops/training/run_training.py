"""Orquesta Fase 4: tunea LightGBM con Optuna (Stratified 5-Fold sobre
train.parquet), refitea el Pipeline ganador sobre todo train.parquet, evalúa una
única vez sobre test.parquet, y loguea+registra el resultado en MLflow
(ADR 0010). Es el entrypoint del stage `train_model` de dvc.yaml:

    uv run python -m churn_mlops.training.run_training
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import optuna
import pandas as pd
from dotenv import load_dotenv

# La consola de Windows usa cp1252 por defecto, que no soporta los emojis que
# mlflow imprime al terminar un run (ej. 🏃). Sin esto, el script crashea después
# de loguear el run exitosamente.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.pipeline import Pipeline

from churn_mlops.config import load_yaml_config
from churn_mlops.training.dataset import load_processed_datasets
from churn_mlops.training.evaluation import compute_classification_metrics
from churn_mlops.training.model_specs import build_lightgbm_pipeline
from churn_mlops.training.tuning import run_optuna_search


def configure_mlflow(training_config: dict[str, Any]) -> None:
    """Carga .env y apunta MLflow al server de DagsHub (nunca hardcoded en src/)."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(training_config["mlflow"]["experiment_name"])


def refit_best_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    high_cardinality_columns: list[str],
    best_params: dict[str, Any],
) -> Pipeline:
    """Arma el Pipeline con los mejores hiperparámetros de Optuna y lo fitea sobre
    TODO train.parquet (no solo los folds de CV)."""
    pipeline = build_lightgbm_pipeline(X_train, high_cardinality_columns, best_params)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline_artifact(pipeline: Pipeline, output_path: str | Path) -> Path:
    """Serializa el Pipeline fiteado a disco vía joblib; DVC lo trackea como out
    del stage train_model (ADR 0010)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def save_best_params(best_params: dict[str, Any], output_path: str | Path) -> Path:
    """Serializa best_params (fixed_params + hiperparámetros de Optuna) a JSON;
    permite a training/run_retrain.py reentrenar reusando el último tuning
    ganador sin volver a correr Optuna (ADR 0012)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    return path


def evaluate_and_log_final_model(
    pipeline: Pipeline,
    best_params: dict[str, Any],
    cv_best_f1: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Evalúa una única vez sobre test.parquet, loguea params/métricas/modelo a
    MLflow y registra la versión en el Model Registry (registered_model_name).
    Devuelve run_id + métricas de test."""
    with mlflow.start_run(run_name="lightgbm-tuned") as run:
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(
            y_test.to_numpy(), y_pred, y_proba
        )

        mlflow.log_params({f"lightgbm__{k}": v for k, v in best_params.items()})
        mlflow.log_param("optuna_n_trials", training_config["optuna"]["n_trials"])
        mlflow.log_param("optuna_cv_folds", training_config["optuna"]["cv_folds"])
        mlflow.log_metric("cv_best_f1", cv_best_f1)
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data_config = load_yaml_config("configs/data.yaml")
    training_config = load_yaml_config("configs/training.yaml")
    configure_mlflow(training_config)

    X_train, X_test, y_train, y_test = load_processed_datasets(data_config)
    high_cardinality_columns = training_config["high_cardinality_columns"]

    study = run_optuna_search(
        X_train, y_train, high_cardinality_columns, training_config
    )
    best_params = {**training_config["fixed_params"], **study.best_params}
    save_best_params(best_params, training_config["best_params_output_path"])

    pipeline = refit_best_pipeline(
        X_train, y_train, high_cardinality_columns, best_params
    )
    model_path = save_pipeline_artifact(pipeline, training_config["model_output_path"])

    result = evaluate_and_log_final_model(
        pipeline, best_params, study.best_value, X_test, y_test, training_config
    )

    cv_folds = training_config["optuna"]["cv_folds"]
    print(f"Mejores hiperparámetros: {best_params}")
    print(f"F1 promedio (CV, {cv_folds}-fold): {study.best_value:.4f}")
    print(f"Métricas en test.parquet: {result}")
    print(f"Pipeline serializado en: {model_path}")


if __name__ == "__main__":
    main()
