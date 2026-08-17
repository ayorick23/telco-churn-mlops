import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from churn_mlops.training.run_training import (
    configure_mlflow,
    evaluate_and_log_final_model,
    refit_best_pipeline,
    save_best_params,
    save_pipeline_artifact,
)

HIGH_CARDINALITY_COLUMNS: list[str] = []
BEST_PARAMS = {
    "n_estimators": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}


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


def test_configure_mlflow_sets_tracking_uri_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    training_config = {"mlflow": {"experiment_name": "test-phase4"}}

    configure_mlflow(training_config)

    assert mlflow.get_tracking_uri() == tracking_uri


def test_refit_best_pipeline_fits_and_predicts() -> None:
    X_train, _, y_train, _ = _sample_data()

    pipeline = refit_best_pipeline(
        X_train, y_train, HIGH_CARDINALITY_COLUMNS, BEST_PARAMS
    )
    predictions = pipeline.predict(X_train)
    probabilities = pipeline.predict_proba(X_train)

    assert len(predictions) == len(X_train)
    assert set(predictions.tolist()) <= {0, 1}
    assert probabilities.shape == (len(X_train), 2)


def test_save_pipeline_artifact_writes_loadable_pickle(tmp_path: Path) -> None:
    X_train, _, y_train, _ = _sample_data()
    pipeline = refit_best_pipeline(
        X_train, y_train, HIGH_CARDINALITY_COLUMNS, BEST_PARAMS
    )
    output_path = tmp_path / "model.pkl"

    saved_path = save_pipeline_artifact(pipeline, output_path)
    loaded_pipeline = joblib.load(saved_path)

    assert saved_path.exists()
    np.testing.assert_array_equal(
        loaded_pipeline.predict(X_train), pipeline.predict(X_train)
    )


def test_save_best_params_writes_readable_json(tmp_path: Path) -> None:
    output_path = tmp_path / "best_params.json"

    saved_path = save_best_params(BEST_PARAMS, output_path)
    loaded = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert loaded == BEST_PARAMS


def test_evaluate_and_log_final_model_logs_run_and_registers_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-phase4-training")

    X_train, X_test, y_train, y_test = _sample_data()
    pipeline = refit_best_pipeline(
        X_train, y_train, HIGH_CARDINALITY_COLUMNS, BEST_PARAMS
    )
    training_config = {
        "registered_model_name": "test-churn-lightgbm",
        "optuna": {"n_trials": 2, "cv_folds": 2},
    }

    result = evaluate_and_log_final_model(
        pipeline, BEST_PARAMS, 0.9, X_test, y_test, training_config
    )

    run = mlflow.get_run(result["run_id"])
    assert any(key.startswith("lightgbm__") for key in run.data.params)
    assert run.data.params["optuna_n_trials"] == "2"
    assert run.data.metrics["cv_best_f1"] == 0.9
    for metric in ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc"]:
        assert metric in run.data.metrics

    versions = mlflow.MlflowClient().search_model_versions("name='test-churn-lightgbm'")
    assert len(versions) == 1
