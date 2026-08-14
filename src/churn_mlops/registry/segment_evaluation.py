"""Evalúa métricas agregadas y por segmento sobre un mismo conjunto de
predicciones ya calculadas — puro, sin MLflow. Reusa
compute_classification_metrics (training/evaluation.py, capa previa en la
cadena Data->Features->Training->Registry/Serving) para no duplicar el
cálculo de F1/precision/recall/etc.

Segmentos = valores de columnas categóricas de baja cardinalidad con impacto
diferencial conocido en churn (EDA notebooks/01_eda.ipynb sección 6.1/8:
Contract, Payment Method, Internet Type — ADR 0012). No es lógica de modelo,
solo de partición de datos ya predichos."""

import numpy as np
import pandas as pd

from churn_mlops.training.evaluation import compute_classification_metrics


def compute_segment_metrics(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    segment_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Para cada columna en segment_columns, para cada valor único no-nulo
    presente en X, calcula compute_classification_metrics restringido a esa
    partición. Clave del dict de salida: "columna=valor" (ej.
    "Contract=Month-to-Month"). X, y_true, y_pred, y_proba deben estar
    alineados por posición (mismo orden de filas). Segmentos donde y_true
    tiene una sola clase presente se saltean — f1/roc_auc no están definidos
    con una sola clase, no es un bug silencioso, queda documentado acá."""
    segment_metrics: dict[str, dict[str, float]] = {}

    for column in segment_columns:
        for value in X[column].dropna().unique():
            mask = (X[column] == value).to_numpy()
            segment_y_true = y_true[mask]

            if len(np.unique(segment_y_true)) < 2:
                continue

            segment_metrics[f"{column}={value}"] = compute_classification_metrics(
                segment_y_true, y_pred[mask], y_proba[mask]
            )

    return segment_metrics
