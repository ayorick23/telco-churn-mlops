"""Orquesta Fase 3: entrena las 4 familias sobre train.parquet/test.parquet, loguea
cada una como run de MLflow separado, y declara la ganadora por
configs/model_selection.yaml:selection_metric. Se corre a mano:

    uv run python -m churn_mlops.training.run_model_selection

Fase 4 decide la familia ganadora y la integra como stage de dvc.yaml (ADR 0009)."""

import io
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv

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
from churn_mlops.training.model_specs import (
    build_catboost_pipeline,
    build_lightgbm_pipeline,
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
)
from churn_mlops.training.preprocessing import compute_scale_pos_weight

_MODEL_FLAVORS: dict[str, ModuleType] = {
    "logistic_regression": mlflow.sklearn,
    "xgboost": mlflow.xgboost,
    "lightgbm": mlflow.lightgbm,
    "catboost": mlflow.catboost,
}


def configure_mlflow(training_config: dict[str, Any]) -> None:
    """Carga .env y apunta MLflow al server de DagsHub (nunca hardcoded en src/)."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(training_config["mlflow"]["experiment_name"])


def build_pipeline_for_family(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_config: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any]]:
    """Arma el Pipeline sin fitear y los params efectivamente usados (incluye
    scale_pos_weight calculado en runtime para XGBoost)."""
    params = dict(training_config["models"][family])
    high_cardinality_columns = training_config["high_cardinality_columns"]

    if family == "logistic_regression":
        pipeline = build_logistic_regression_pipeline(
            X_train, high_cardinality_columns, params
        )
    elif family == "xgboost":
        params["scale_pos_weight"] = compute_scale_pos_weight(y_train)
        pipeline = build_xgboost_pipeline(X_train, high_cardinality_columns, params)
    elif family == "lightgbm":
        pipeline = build_lightgbm_pipeline(X_train, high_cardinality_columns, params)
    elif family == "catboost":
        pipeline = build_catboost_pipeline(X_train, params)
    else:
        raise ValueError(f"Familia de modelo desconocida: {family}")
    return pipeline, params


def train_and_log_family(
    family: str,
    pipeline: Pipeline,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Fitea, evalúa y loguea un run de MLflow: preprocesador vía mlflow.sklearn
    (siempre es sklearn, sin importar la familia) y el modelo final vía su flavor
    nativo. Devuelve métricas + run_id para la tabla comparativa."""
    model_flavor = _MODEL_FLAVORS[family]

    with mlflow.start_run(run_name=family) as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(y_test.to_numpy(), y_pred, y_proba)

        mlflow.log_params({f"{family}__{k}": v for k, v in params.items()})
        mlflow.log_metrics(metrics)
        # serialization_format="pickle": el default (skops) rechaza tipos internos
        # de ColumnTransformer/OneHotEncoder (ej. numpy.dtype) como "no confiables".
        # Es nuestro propio modelo, no uno de terceros, así que pickle es adecuado.
        mlflow.sklearn.log_model(
            pipeline.named_steps["preprocess"],
            artifact_path="preprocessor",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        model_flavor.log_model(pipeline.named_steps["model"], artifact_path="model")

        return {"family": family, "run_id": run.info.run_id, **metrics}


def build_comparison_table(
    results: list[dict[str, Any]], selection_metric: str
) -> pd.DataFrame:
    """Ordena las familias por la métrica de selección, la mejor primero."""
    return (
        pd.DataFrame(results)
        .sort_values(selection_metric, ascending=False)
        .reset_index(drop=True)
    )


def _table_to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def save_comparison_report(
    table: pd.DataFrame, selection_metric: str, output_dir: str | Path
) -> tuple[Path, Path]:
    """Guarda la tabla comparativa (CSV) y el veredicto (Markdown) en reports/."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "model_selection_results.csv"
    table.to_csv(csv_path, index=False)

    winner = table.iloc[0]
    summary_path = output_path / "model_selection_summary.md"
    summary_path.write_text(
        "# Fase 3 — Selección de modelo\n\n"
        f"Métrica de selección: **{selection_metric}**\n\n"
        f"Familia ganadora: **{winner['family']}** "
        f"({selection_metric}={winner[selection_metric]:.4f})\n\n"
        f"{_table_to_markdown(table)}\n",
        encoding="utf-8",
    )
    return csv_path, summary_path


def main() -> None:
    data_config = load_yaml_config("configs/data.yaml")
    training_config = load_yaml_config("configs/model_selection.yaml")
    configure_mlflow(training_config)

    X_train, X_test, y_train, y_test = load_processed_datasets(data_config)

    results = []
    for family in training_config["models"]:
        pipeline, params = build_pipeline_for_family(
            family, X_train, y_train, training_config
        )
        result = train_and_log_family(
            family, pipeline, params, X_train, X_test, y_train, y_test
        )
        results.append(result)
        print(f"{family}: {result}")

    selection_metric = training_config["selection_metric"]
    table = build_comparison_table(results, selection_metric)
    csv_path, summary_path = save_comparison_report(
        table, selection_metric, training_config["reports_dir"]
    )

    with mlflow.start_run(run_name="summary"):
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_param("winning_family", table.iloc[0]["family"])
        mlflow.log_metric(f"best_{selection_metric}", table.iloc[0][selection_metric])

    print(table.to_string(index=False))
    print(f"\nGanador: {table.iloc[0]['family']}")


if __name__ == "__main__":
    main()
