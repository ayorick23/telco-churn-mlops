"""Lógica pura de decisión de reentrenamiento a partir del resumen cross-batch
de Fase 5 (reports/monitoring/drift_monitoring_results.csv). Sin I/O, sin
MLflow — testeable en aislamiento, mismo rol que drift_simulation.py/
evaluation.py en fases previas.

No decide qué tan sensible es la detección de Evidently (drift_share,
configs/monitoring.yaml, ADR 0011) — decide si el nivel de drift YA detectado
amerita la ACCIÓN de reentrenar (configs/retraining.yaml, ADR 0012)."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RetrainDecision:
    should_retrain: bool
    trigger_metric: str
    observed_value: float
    threshold: float
    triggering_batch: str


def evaluate_retrain_trigger(
    drift_summary: pd.DataFrame, retraining_config: dict[str, Any]
) -> RetrainDecision:
    """should_retrain = True si el MÁXIMO de trigger_metric en todo el resumen
    supera retrain_threshold. Se usa el máximo (no la última fila) porque el
    CSV de Fase 5 es un barrido sintético completo por intensidad creciente,
    no una serie temporal de batches de producción — el máximo es robusto al
    orden del CSV y no ignora un batch intermedio severo."""
    trigger_metric = retraining_config["trigger_metric"]
    threshold = retraining_config["retrain_threshold"]

    worst_row = drift_summary.loc[drift_summary[trigger_metric].idxmax()]
    observed_value = float(worst_row[trigger_metric])

    return RetrainDecision(
        should_retrain=observed_value > threshold,
        trigger_metric=trigger_metric,
        observed_value=observed_value,
        threshold=threshold,
        triggering_batch=str(worst_row["batch_name"]),
    )
