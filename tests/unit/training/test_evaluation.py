import numpy as np
import pytest

from churn_mlops.training.evaluation import compute_classification_metrics


def test_compute_classification_metrics_perfect_classifier() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.9, 1.0])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    for value in metrics.values():
        assert value == pytest.approx(1.0)


def test_compute_classification_metrics_with_known_errors() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.7, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    # TP=2, FP=1, FN=0 -> precision=2/3, recall=1.0, f1=2*(2/3*1)/(2/3+1)=0.8
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["accuracy"] == pytest.approx(0.75)
