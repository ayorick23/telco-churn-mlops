"""Métricas de evaluación — puras, reciben predicciones ya calculadas (sin MLflow,
sin acoplarse a un framework de modelo en particular)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """F1/precision/recall de la clase positiva (churn=1, métrica de selección de
    configs/model_selection.yaml) + accuracy, ROC-AUC, PR-AUC como secundarias."""
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }
