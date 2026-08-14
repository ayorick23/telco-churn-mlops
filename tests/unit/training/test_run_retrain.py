import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from churn_mlops.training.run_retrain import (
    evaluate_and_log_refit_model,
    load_best_params,
)
from churn_mlops.training.run_training import refit_best_pipeline

HIGH_CARDINALITY_COLUMNS: list[str] = []
FIXED_PARAMS = {"class_weight": "balanced", "random_state": 42, "verbose": -1}
PERSISTED_PARAMS = {**FIXED_PARAMS, "n_estimators": 10}


def _sample_data(
    n: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "Age": rng.integers(18, 80, size=n),
            "Contract": rng.choice(["Month-to-Month", "One Year"], size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="Churn Label")
    y.iloc[0] = 0
    y.iloc[1] = 1
    return X.iloc[:20], X.iloc[20:], y.iloc[:20], y.iloc[20:]


def test_load_best_params_merges_fixed_and_persisted(tmp_path: Path) -> None:
    params_path = tmp_path / "best_params.json"
    params_path.write_text(json.dumps({"n_estimators": 10}), encoding="utf-8")

    result = load_best_params(params_path, FIXED_PARAMS)

    assert result == PERSISTED_PARAMS


def test_load_best_params_raises_with_explicit_message_when_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError, match="train_model"):
        load_best_params(missing_path, FIXED_PARAMS)


def test_evaluate_and_log_refit_model_logs_run_without_cv_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-phase6-retrain")

    X_train, X_test, y_train, y_test = _sample_data()
    pipeline = refit_best_pipeline(
        X_train, y_train, HIGH_CARDINALITY_COLUMNS, PERSISTED_PARAMS
    )
    training_config = {"registered_model_name": "test-churn-lightgbm-retrain"}

    result = evaluate_and_log_refit_model(
        pipeline, PERSISTED_PARAMS, X_test, y_test, training_config
    )

    run = mlflow.get_run(result["run_id"])
    assert run.info.run_name == "lightgbm-reused-params"
    assert any(key.startswith("lightgbm__") for key in run.data.params)
    assert "cv_best_f1" not in run.data.metrics
    for metric in ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc"]:
        assert metric in run.data.metrics

    versions = mlflow.MlflowClient().search_model_versions(
        "name='test-churn-lightgbm-retrain'"
    )
    assert len(versions) == 1
