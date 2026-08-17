import pandas as pd

from churn_mlops.monitoring.retrain_trigger import evaluate_retrain_trigger

RETRAINING_CONFIG = {"trigger_metric": "dataset_drift_share", "retrain_threshold": 0.10}


def _drift_summary(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch_name": [f"batch_intensity_{i:.2f}" for i in range(len(values))],
            "dataset_drift_share": values,
        }
    )


def test_no_retrain_when_all_values_below_threshold() -> None:
    drift_summary = _drift_summary([0.026, 0.051, 0.08])

    decision = evaluate_retrain_trigger(drift_summary, RETRAINING_CONFIG)

    assert decision.should_retrain is False
    assert decision.observed_value == 0.08


def test_retrain_when_max_value_exceeds_threshold() -> None:
    drift_summary = _drift_summary([0.026, 0.051, 0.128, 0.128, 0.128])

    decision = evaluate_retrain_trigger(drift_summary, RETRAINING_CONFIG)

    assert decision.should_retrain is True
    assert decision.observed_value == 0.128


def test_triggering_batch_is_the_row_with_the_max_value_not_the_last_row() -> None:
    drift_summary = pd.DataFrame(
        {
            "batch_name": ["batch_a", "batch_b", "batch_c"],
            "dataset_drift_share": [0.128, 0.5, 0.03],
        }
    )

    decision = evaluate_retrain_trigger(drift_summary, RETRAINING_CONFIG)

    assert decision.triggering_batch == "batch_b"
    assert decision.observed_value == 0.5
