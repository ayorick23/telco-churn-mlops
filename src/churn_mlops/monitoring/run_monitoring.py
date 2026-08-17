"""Orquesta Fase 5: genera N batches sintéticos a intensidad creciente
(configs/monitoring.yaml:batch_generation.intensity_schedule), corre detección de
drift con Evidently contra train.parquet como referencia, loguea un run de MLflow
por batch + un run de resumen, y guarda reportes en reports_dir (ADR 0011). Se
corre a mano (no es stage de dvc.yaml):

    uv run python -m churn_mlops.monitoring.run_monitoring
"""

import io
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv

# La consola de Windows usa cp1252 por defecto, que no soporta los emojis que
# mlflow imprime al terminar un run (ej. 🏃). Sin esto, el script crashea después
# de loguear el run exitosamente. El isinstance narrowa sys.stdout a
# TextIOWrapper para mypy (TextIOBase no declara reconfigure).
if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from churn_mlops.config import load_yaml_config
from churn_mlops.data.drift_simulation import generate_drifted_batch
from churn_mlops.monitoring.drift_detection import (
    run_drift_report,
    save_drift_report,
    summarize_drift_result,
)


def configure_mlflow(monitoring_config: dict[str, Any]) -> None:
    """Carga .env y apunta MLflow al server de DagsHub — idéntico patrón a
    run_training.py/run_model_selection.py."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(monitoring_config["mlflow"]["experiment_name"])


def load_reference(data_config: dict[str, Any]) -> pd.DataFrame:
    """Lee train.parquet SIN el target: es la referencia contra la que Evidently
    compara cada batch sintético."""
    processed_dir = Path(data_config["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    return train_df.drop(columns=[data_config["target_column"]])


def run_batch(
    batch_name: str,
    intensity: float,
    reference_df: pd.DataFrame,
    monitoring_config: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    """Genera un batch a la intensidad dada, corre detección de drift, loguea un
    run de MLflow (params: batch_name/intensity/drift_share; metrics: resumen;
    artifacts: HTML+JSON) y devuelve una fila de resumen para la tabla final."""
    drift_share = monitoring_config["drift_share"]

    batch_df = generate_drifted_batch(
        reference_df,
        monitoring_config["batch_generation"],
        intensity=intensity,
        target_column=target_column,
    )

    result = run_drift_report(reference_df, batch_df, drift_share)
    summary = summarize_drift_result(result)
    html_path, json_path = save_drift_report(
        result, monitoring_config["reports_dir"], batch_name
    )

    with mlflow.start_run(run_name=batch_name) as run:
        mlflow.log_param("batch_name", batch_name)
        mlflow.log_param("intensity", intensity)
        mlflow.log_param("drift_share_threshold", drift_share)
        mlflow.log_metrics(summary)
        mlflow.log_artifact(str(html_path))
        mlflow.log_artifact(str(json_path))
        run_id = run.info.run_id

    return {
        "batch_name": batch_name,
        "intensity": intensity,
        "run_id": run_id,
        **summary,
    }


def build_summary_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Arma la tabla cross-batch ordenada por intensity ascendente."""
    return pd.DataFrame(results).sort_values("intensity").reset_index(drop=True)


def _table_to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def save_summary_report(
    table: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path]:
    """Guarda la tabla cross-batch (CSV) y un resumen legible (Markdown) en
    reports_dir — mismo patrón que save_comparison_report en
    run_model_selection.py."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "drift_monitoring_results.csv"
    table.to_csv(csv_path, index=False)

    summary_path = output_path / "drift_monitoring_summary.md"
    summary_path.write_text(
        f"# Fase 5 — Simulación de drift y monitoreo\n\n{_table_to_markdown(table)}\n",
        encoding="utf-8",
    )
    return csv_path, summary_path


def main() -> None:
    data_config = load_yaml_config("configs/data.yaml")
    monitoring_config = load_yaml_config("configs/monitoring.yaml")
    configure_mlflow(monitoring_config)

    reference_df = load_reference(data_config)
    target_column = data_config["target_column"]

    results = []
    for intensity in monitoring_config["batch_generation"]["intensity_schedule"]:
        batch_name = f"batch_intensity_{intensity:.2f}"
        result = run_batch(
            batch_name, intensity, reference_df, monitoring_config, target_column
        )
        results.append(result)
        print(f"{batch_name}: {result}")

    table = build_summary_table(results)
    csv_path, summary_path = save_summary_report(
        table, monitoring_config["reports_dir"]
    )

    with mlflow.start_run(run_name="summary"):
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(summary_path))

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
