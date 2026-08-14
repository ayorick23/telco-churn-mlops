"""Wrapper delgado sobre Evidently 0.7.x (Report + DataDriftPreset) — Fase 5.
El detalle por columna vive en el HTML/JSON guardado por save_drift_report; acá
solo se extrae el resumen agregado (share de columnas con drift) que se loguea
como métrica de MLflow. No hay lógica de trigger/retraining acá (ver ADR 0011,
eso es Fase 6)."""

from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset


def build_drift_report(drift_share: float) -> Report:
    """Arma el Report de Evidently con un único preset (DataDriftPreset),
    configurado con el drift_share de configs/monitoring.yaml."""
    return Report(metrics=[DataDriftPreset(drift_share=drift_share)])


def run_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, drift_share: float
) -> Snapshot:
    """Corre el Report: reference_df = train.parquet (sin target), current_df =
    un batch sintético (sin target). Devuelve el Snapshot crudo de Evidently."""
    report = build_drift_report(drift_share)
    return report.run(
        current_data=Dataset.from_pandas(current_df),
        reference_data=Dataset.from_pandas(reference_df),
    )


def summarize_drift_result(result: Snapshot) -> dict[str, float]:
    """Extrae de Snapshot.dict()['metrics'] únicamente el resumen agregado:
    dataset_drift_share (0-1) y n_drifted_columns (conteo). Busca la entrada
    cuyo metric_name empieza con 'DriftedColumnsCount' (el nombre completo
    incluye el propio drift_share, ej. 'DriftedColumnsCount(drift_share=0.5)',
    por eso startswith y no ==). El detalle por columna (ValueDrift por cada
    feature) NO se parsea acá — vive intacto en el HTML/JSON guardado por
    save_drift_report, evita duplicar parsing frágil ante upgrades de la
    librería."""
    metrics = result.dict()["metrics"]
    drifted_columns_metric = next(
        m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount")
    )
    return {
        "n_drifted_columns": float(drifted_columns_metric["value"]["count"]),
        "dataset_drift_share": float(drifted_columns_metric["value"]["share"]),
    }


def save_drift_report(
    result: Snapshot, output_dir: str | Path, batch_name: str
) -> tuple[Path, Path]:
    """Guarda el reporte completo como HTML + JSON en output_dir/{batch_name}.
    {html,json} — detalle por columna para inspección manual, no se re-parsea
    en Python."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    html_path = output_path / f"{batch_name}.html"
    json_path = output_path / f"{batch_name}.json"
    result.save_html(str(html_path))
    result.save_json(str(json_path))
    return html_path, json_path
