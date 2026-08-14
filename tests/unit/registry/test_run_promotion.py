from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from churn_mlops.registry.comparison import decide_promotion
from churn_mlops.registry.mlflow_client import (
    get_champion_version,
    load_pipeline_from_run,
    promote_challenger,
)
from churn_mlops.registry.run_promotion import (
    configure_mlflow,
    evaluate_pipeline,
    save_promotion_report,
)
from churn_mlops.training.model_specs import build_lightgbm_pipeline

MODEL_NAME = "test-churn-lightgbm-promotion"
REGISTRY_CONFIG = {
    "promotion_metric": "f1",
    "promotion_margin": 0.01,
    "segment_columns": ["Contract"],
    "segment_regression_tolerance": 0.03,
}


def _dataset(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """`signal` determina el target casi perfectamente — permite fitear un
    modelo "bueno" (fitea sobre y real) y uno "malo" (fitea sobre y barajado,
    sin relación con signal) con un gap de F1 grande y reproducible."""
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, n)
    X = pd.DataFrame(
        {
            "signal": signal,
            "Contract": rng.choice(["Month-to-Month", "One Year"], size=n),
        }
    )
    y = pd.Series((signal > 0).astype(int), name="Churn Label")
    return X, y


def _log_and_register(model_name: str, X: pd.DataFrame, y_for_fit: pd.Series) -> str:
    pipeline = build_lightgbm_pipeline(
        X, [], {"n_estimators": 20, "random_state": 42, "verbose": -1}
    )
    pipeline.fit(X, y_for_fit)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            pipeline.named_steps["preprocess"],
            artifact_path="preprocessor",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        model_info = mlflow.lightgbm.log_model(
            pipeline.named_steps["model"],
            artifact_path="model",
            registered_model_name=model_name,
        )
    return model_info.registered_model_version


@pytest.fixture
def mlflow_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> mlflow.MlflowClient:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-phase6-promotion")
    return mlflow.MlflowClient()


def test_bootstrap_promotes_without_comparing(
    mlflow_client: mlflow.MlflowClient,
) -> None:
    X, y = _dataset()
    challenger_version = _log_and_register(MODEL_NAME, X, y)

    champion = get_champion_version(mlflow_client, MODEL_NAME, "champion")
    assert champion is None  # caso bootstrap

    promote_challenger(mlflow_client, MODEL_NAME, "champion", challenger_version)

    result = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert result.version == challenger_version


def test_challenger_wins_and_gets_promoted(mlflow_client: mlflow.MlflowClient) -> None:
    X, y = _dataset()
    y_shuffled = pd.Series(
        np.random.default_rng(1).permutation(y.to_numpy()), name=y.name
    )

    champion_version = _log_and_register(MODEL_NAME, X, y_shuffled)  # modelo débil
    mlflow_client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
    challenger_version = _log_and_register(MODEL_NAME, X, y)  # modelo bueno

    champion_model_version = mlflow_client.get_model_version(
        MODEL_NAME, champion_version
    )
    challenger_model_version = mlflow_client.get_model_version(
        MODEL_NAME, challenger_version
    )

    champion_pipeline = load_pipeline_from_run(champion_model_version.run_id)
    challenger_pipeline = load_pipeline_from_run(challenger_model_version.run_id)

    champion_metrics, champion_segments = evaluate_pipeline(
        champion_pipeline, X, y, REGISTRY_CONFIG["segment_columns"]
    )
    challenger_metrics, challenger_segments = evaluate_pipeline(
        challenger_pipeline, X, y, REGISTRY_CONFIG["segment_columns"]
    )

    decision = decide_promotion(
        champion_metrics["f1"],
        challenger_metrics["f1"],
        {k: v["f1"] for k, v in champion_segments.items()},
        {k: v["f1"] for k, v in challenger_segments.items()},
        REGISTRY_CONFIG,
    )
    assert decision.promote is True

    promote_challenger(mlflow_client, MODEL_NAME, "champion", challenger_version)

    result = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert result.version == challenger_version


def test_challenger_loses_and_alias_is_unchanged(
    mlflow_client: mlflow.MlflowClient,
) -> None:
    X, y = _dataset()
    y_shuffled = pd.Series(
        np.random.default_rng(1).permutation(y.to_numpy()), name=y.name
    )

    champion_version = _log_and_register(MODEL_NAME, X, y)  # modelo bueno
    mlflow_client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
    challenger_version = _log_and_register(MODEL_NAME, X, y_shuffled)  # modelo débil

    champion_model_version = mlflow_client.get_model_version(
        MODEL_NAME, champion_version
    )
    challenger_model_version = mlflow_client.get_model_version(
        MODEL_NAME, challenger_version
    )
    champion_pipeline = load_pipeline_from_run(champion_model_version.run_id)
    challenger_pipeline = load_pipeline_from_run(challenger_model_version.run_id)

    champion_metrics, champion_segments = evaluate_pipeline(
        champion_pipeline, X, y, REGISTRY_CONFIG["segment_columns"]
    )
    challenger_metrics, challenger_segments = evaluate_pipeline(
        challenger_pipeline, X, y, REGISTRY_CONFIG["segment_columns"]
    )

    decision = decide_promotion(
        champion_metrics["f1"],
        challenger_metrics["f1"],
        {k: v["f1"] for k, v in champion_segments.items()},
        {k: v["f1"] for k, v in challenger_segments.items()},
        REGISTRY_CONFIG,
    )
    assert decision.promote is False

    # alias NO se mueve porque no se llama a promote_challenger en este caso
    result = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert result.version == champion_version


def test_configure_mlflow_sets_tracking_uri_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    registry_config = {"mlflow": {"experiment_name": "test-phase6-promotion-config"}}

    configure_mlflow(registry_config)

    assert mlflow.get_tracking_uri() == tracking_uri


def test_save_promotion_report_bootstrap_writes_csv_and_md(
    mlflow_client: mlflow.MlflowClient, tmp_path: Path
) -> None:
    X, y = _dataset()
    version = _log_and_register(MODEL_NAME, X, y)
    challenger_version = mlflow_client.get_model_version(MODEL_NAME, version)
    challenger_metrics, _ = evaluate_pipeline(
        load_pipeline_from_run(challenger_version.run_id),
        X,
        y,
        REGISTRY_CONFIG["segment_columns"],
    )
    output_dir = tmp_path / "reports"

    csv_path, summary_path = save_promotion_report(
        challenger_version, None, challenger_metrics, None, None, output_dir
    )

    assert csv_path.exists()
    assert summary_path.exists()
    assert "Bootstrap" in summary_path.read_text(encoding="utf-8")
    rows = pd.read_csv(csv_path)
    assert list(rows["role"]) == ["challenger"]


def test_save_promotion_report_with_decision_includes_both_rows(
    mlflow_client: mlflow.MlflowClient, tmp_path: Path
) -> None:
    X, y = _dataset()
    challenger_version = mlflow_client.get_model_version(
        MODEL_NAME, _log_and_register(MODEL_NAME, X, y)
    )
    champion_version = mlflow_client.get_model_version(
        MODEL_NAME, _log_and_register(MODEL_NAME, X, y)
    )
    metrics, _ = evaluate_pipeline(
        load_pipeline_from_run(challenger_version.run_id),
        X,
        y,
        REGISTRY_CONFIG["segment_columns"],
    )
    decision = decide_promotion(0.5, 0.6, {}, {}, REGISTRY_CONFIG)
    output_dir = tmp_path / "reports"

    csv_path, summary_path = save_promotion_report(
        challenger_version, champion_version, metrics, metrics, decision, output_dir
    )

    rows = pd.read_csv(csv_path)
    assert set(rows["role"]) == {"challenger", "champion"}
    assert "Promovido" in summary_path.read_text(encoding="utf-8")
