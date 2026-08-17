from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from churn_mlops.training.run_model_selection import (
    build_comparison_table,
    build_pipeline_for_family,
    configure_mlflow,
    save_comparison_report,
    train_and_log_family,
)


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
    training_config = {"mlflow": {"experiment_name": "test-experiment"}}

    configure_mlflow(training_config)

    assert mlflow.get_tracking_uri() == tracking_uri


def test_build_comparison_table_sorts_by_selection_metric_descending() -> None:
    results = [
        {"family": "logistic_regression", "f1": 0.5},
        {"family": "catboost", "f1": 0.8},
        {"family": "xgboost", "f1": 0.65},
    ]

    table = build_comparison_table(results, selection_metric="f1")

    assert table["family"].tolist() == ["catboost", "xgboost", "logistic_regression"]


def test_save_comparison_report_writes_csv_and_markdown_with_winner(
    tmp_path: Path,
) -> None:
    table = build_comparison_table(
        [
            {"family": "catboost", "f1": 0.8},
            {"family": "logistic_regression", "f1": 0.5},
        ],
        selection_metric="f1",
    )

    csv_path, summary_path = save_comparison_report(table, "f1", tmp_path)

    assert csv_path.exists()
    assert summary_path.exists()
    assert "catboost" in summary_path.read_text(encoding="utf-8")


def test_train_and_log_family_logs_run_with_expected_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-logistic-regression")

    X_train, X_test, y_train, y_test = _sample_data()
    training_config = {
        "high_cardinality_columns": [],
        "models": {"logistic_regression": {"max_iter": 200}},
    }
    pipeline, params = build_pipeline_for_family(
        "logistic_regression", X_train, y_train, training_config
    )

    result = train_and_log_family(
        "logistic_regression", pipeline, params, X_train, X_test, y_train, y_test
    )

    run = mlflow.get_run(result["run_id"])
    assert any(key.startswith("logistic_regression__") for key in run.data.params)
    for metric in ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc"]:
        assert metric in run.data.metrics
