"""Simulación de drift (Fase 5): perturba muestras de train.parquet para generar
batches sintéticos que imitan datos futuros con drift. Puro y determinístico
(random_state fijo) — sin MLflow/Evidently/IO, ver monitoring/ para eso. Nunca
perturba la columna target (ej. Churn Label): ver ADR 0006/0011."""

from typing import Any

import numpy as np
import pandas as pd


def sample_reference_rows(
    reference_df: pd.DataFrame, batch_size: int, random_state: int
) -> pd.DataFrame:
    """Muestrea `batch_size` filas de reference_df con reemplazo (simula un
    batch de scoring). Devuelve una copia con índice reseteado."""
    return reference_df.sample(
        n=batch_size, replace=True, random_state=random_state
    ).reset_index(drop=True)


def apply_numeric_shift(
    series: pd.Series,
    mean_shift: float,
    std_shift: float,
    intensity: float,
    clip_min: float | None = None,
) -> pd.Series:
    """Suma intensity * mean_shift a cada valor y escala la desviación respecto
    a la media original en (1 + intensity * std_shift) (std_shift=0 => solo
    shift de media, sin recaling de dispersión). Aplica clip_min si se
    especifica (ej. Tenure no puede ser negativo)."""
    mean = series.mean()
    shifted = mean + (series - mean) * (1 + intensity * std_shift)
    shifted = shifted + intensity * mean_shift
    if clip_min is not None:
        shifted = shifted.clip(lower=clip_min)
    return shifted


def apply_categorical_reweight(
    series: pd.Series,
    target_distribution: dict[str, float],
    intensity: float,
    random_state: int,
) -> pd.Series:
    """Re-samplea la columna completa desde una distribución interpolada
    linealmente entre la distribución empírica de `series` (a intensity=0) y
    target_distribution (a intensity=1). Devuelve una Series nueva del mismo
    largo e índice que la original."""
    empirical = series.value_counts(normalize=True)
    categories = sorted(set(empirical.index) | set(target_distribution))

    probs = np.array(
        [
            (1 - intensity) * empirical.get(category, 0.0)
            + intensity * target_distribution.get(category, 0.0)
            for category in categories
        ]
    )
    probs = probs / probs.sum()

    rng = np.random.default_rng(random_state)
    values = rng.choice(categories, size=len(series), p=probs)
    return pd.Series(values, index=series.index, name=series.name)


def apply_drift_specs(
    batch_df: pd.DataFrame,
    drift_specs: dict[str, dict[str, Any]],
    intensity: float,
    random_state: int,
    target_column: str,
) -> pd.DataFrame:
    """Despacha cada entrada de drift_specs (configs/monitoring.yaml) a
    apply_numeric_shift/apply_categorical_reweight según spec['type'], mismo
    patrón que tuning.py::build_search_space. Lanza ValueError si drift_specs
    contiene target_column (nunca se perturba el target, ADR 0006/0011) o un
    `type` desconocido."""
    if target_column in drift_specs:
        raise ValueError(
            f"drift_specs no puede perturbar la columna target ({target_column})"
        )

    batch_df = batch_df.copy()
    for column, spec in drift_specs.items():
        if spec["type"] == "numeric_shift":
            batch_df[column] = apply_numeric_shift(
                batch_df[column],
                mean_shift=spec["mean_shift"],
                std_shift=spec.get("std_shift", 0.0),
                intensity=intensity,
                clip_min=spec.get("clip_min"),
            )
        elif spec["type"] == "categorical_reweight":
            batch_df[column] = apply_categorical_reweight(
                batch_df[column],
                target_distribution=spec["target_distribution"],
                intensity=intensity,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Tipo de drift desconocido: {spec['type']}")
    return batch_df


def generate_drifted_batch(
    reference_df: pd.DataFrame,
    drift_config: dict[str, Any],
    intensity: float,
    target_column: str,
) -> pd.DataFrame:
    """Orquesta: sample_reference_rows + apply_drift_specs. `drift_config` es
    configs/monitoring.yaml:batch_generation completo (batch_size, random_state,
    drift_specs). `intensity` viene del intensity_schedule del caller — no se
    lee de acá, para que run_monitoring.py controle el barrido explícitamente."""
    random_state = drift_config["random_state"]
    batch_df = sample_reference_rows(
        reference_df, drift_config["batch_size"], random_state
    )
    return apply_drift_specs(
        batch_df,
        drift_config["drift_specs"],
        intensity,
        random_state,
        target_column,
    )
