import numpy as np
import pandas as pd

from churn_mlops.registry.segment_evaluation import compute_segment_metrics


def test_partitions_correctly_by_categorical_column_value() -> None:
    X = pd.DataFrame({"Contract": ["Month-to-Month"] * 4 + ["One Year"] * 4})
    y_true = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9])

    result = compute_segment_metrics(X, y_true, y_pred, y_proba, ["Contract"])

    assert set(result.keys()) == {"Contract=Month-to-Month", "Contract=One Year"}
    assert result["Contract=Month-to-Month"]["f1"] == 1.0


def test_single_class_segment_is_skipped_without_raising() -> None:
    X = pd.DataFrame({"Contract": ["Month-to-Month"] * 3 + ["Two Year"] * 3})
    y_true = np.array([0, 1, 0, 0, 0, 0])  # Two Year segment: only class 0
    y_pred = np.array([0, 1, 0, 0, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.1, 0.1, 0.1, 0.1])

    result = compute_segment_metrics(X, y_true, y_pred, y_proba, ["Contract"])

    assert "Contract=Month-to-Month" in result
    assert "Contract=Two Year" not in result


def test_output_keys_use_column_equals_value_format() -> None:
    X = pd.DataFrame({"Payment Method": ["Credit Card"] * 4})
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.1, 0.9])

    result = compute_segment_metrics(X, y_true, y_pred, y_proba, ["Payment Method"])

    assert list(result.keys()) == ["Payment Method=Credit Card"]
