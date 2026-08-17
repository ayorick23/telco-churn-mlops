from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from churn_mlops.registry.comparison import decide_promotion
from churn_mlops.registry.mlflow_client import get_champion_version, promote_challenger
from churn_mlops.registry.run_promotion import (
    configure_mlflow,
    evaluate_pipeline,
    load_local_pipeline,
    save_promotion_report,
)
from churn_mlops.training.model_specs import build_lightgbm_pipeline
from churn_mlops.training.run_training import save_pipeline_artifact

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


def _fit_pipeline(X: pd.DataFrame, y_for_fit: pd.Series):
    pipeline = build_lightgbm_pipeline(
        X, [], {"n_estimators": 20, "random_state": 42, "verbose": -1}
    )
    pipeline.fit(X, y_for_fit)
    return pipeline


def _register_version(model_name: str, pipeline) -> str:
    """Registra una versión nueva en el Model Registry (metadata: versión +
    disponibilidad para alias). Ya no hace falta loguear artifacts
    descargables — run_promotion.py lee los bytes del pipeline de disco, no
    de MLflow (ADR 0012)."""
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
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


def test_load_local_pipeline_round_trips(tmp_path: Path) -> None:
    X, y = _dataset()
    pipeline = _fit_pipeline(X, y)
    path = tmp_path / "pipeline.pkl"
    save_pipeline_artifact(pipeline, path)

    loaded = load_local_pipeline(path)

    np.testing.assert_array_equal(loaded.predict(X), pipeline.predict(X))


def test_bootstrap_promotes_without_comparing(
    mlflow_client: mlflow.MlflowClient, tmp_path: Path
) -> None:
    X, y = _dataset()
    pipeline = _fit_pipeline(X, y)
    challenger_version = _register_version(MODEL_NAME, pipeline)
    champion_path = tmp_path / "champion.pkl"

    champion = get_champion_version(mlflow_client, MODEL_NAME, "champion")
    assert champion is None  # caso bootstrap

    promote_challenger(mlflow_client, MODEL_NAME, "champion", challenger_version)
    save_pipeline_artifact(pipeline, champion_path)

    result = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert result.version == challenger_version
    np.testing.assert_array_equal(
        load_local_pipeline(champion_path).predict(X), pipeline.predict(X)
    )


def test_challenger_wins_and_gets_promoted(
    mlflow_client: mlflow.MlflowClient, tmp_path: Path
) -> None:
    X, y = _dataset()
    y_shuffled = pd.Series(
        np.random.default_rng(1).permutation(y.to_numpy()), name=y.name
    )
    weak_pipeline = _fit_pipeline(X, y_shuffled)  # champion débil
    strong_pipeline = _fit_pipeline(X, y)  # challenger bueno

    champion_version = _register_version(MODEL_NAME, weak_pipeline)
    mlflow_client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
    champion_path = tmp_path / "champion.pkl"
    save_pipeline_artifact(weak_pipeline, champion_path)

    challenger_version = _register_version(MODEL_NAME, strong_pipeline)

    champion_metrics, champion_segments = evaluate_pipeline(
        load_local_pipeline(champion_path), X, y, REGISTRY_CONFIG["segment_columns"]
    )
    challenger_metrics, challenger_segments = evaluate_pipeline(
        strong_pipeline, X, y, REGISTRY_CONFIG["segment_columns"]
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
    save_pipeline_artifact(strong_pipeline, champion_path)

    result = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert result.version == challenger_version
    np.testing.assert_array_equal(
        load_local_pipeline(champion_path).predict(X), strong_pipeline.predict(X)
    )


def test_challenger_loses_and_alias_and_file_are_unchanged(
    mlflow_client: mlflow.MlflowClient, tmp_path: Path
) -> None:
    X, y = _dataset()
    y_shuffled = pd.Series(
        np.random.default_rng(1).permutation(y.to_numpy()), name=y.name
    )
    strong_pipeline = _fit_pipeline(X, y)  # champion bueno
    weak_pipeline = _fit_pipeline(X, y_shuffled)  # challenger débil

    champion_version = _register_version(MODEL_NAME, strong_pipeline)
    mlflow_client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
    champion_path = tmp_path / "champion.pkl"
    save_pipeline_artifact(strong_pipeline, champion_path)

    _register_version(MODEL_NAME, weak_pipeline)

    champion_metrics, champion_segments = evaluate_pipeline(
        load_local_pipeline(champion_path), X, y, REGISTRY_CONFIG["segment_columns"]
    )
    challenger_metrics, challenger_segments = evaluate_pipeline(
        weak_pipeline, X, y, REGISTRY_CONFIG["segment_columns"]
    )

    decision = decide_promotion(
        champion_metrics["f1"],
        challenger_metrics["f1"],
        {k: v["f1"] for k, v in champion_segments.items()},
        {k: v["f1"] for k, v in challenger_segments.items()},
        REGISTRY_CONFIG,
    )
    assert decision.promote is False

    # ni el alias ni el archivo local del champion se tocan en este caso
    result = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert result.version == champion_version
    np.testing.assert_array_equal(
        load_local_pipeline(champion_path).predict(X), strong_pipeline.predict(X)
    )


def test_configure_mlflow_sets_tracking_uri_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    registry_config = {"mlflow": {"experiment_name": "test-phase6-promotion-config"}}

    configure_mlflow(registry_config)

    assert mlflow.get_tracking_uri() == tracking_uri


def test_save_promotion_report_bootstrap_writes_csv_and_md(tmp_path: Path) -> None:
    X, y = _dataset()
    pipeline = _fit_pipeline(X, y)
    challenger_path = tmp_path / "challenger.pkl"
    save_pipeline_artifact(pipeline, challenger_path)
    challenger_metrics, _ = evaluate_pipeline(
        load_local_pipeline(challenger_path), X, y, REGISTRY_CONFIG["segment_columns"]
    )

    class _FakeVersion:
        version = "1"

    output_dir = tmp_path / "reports"
    csv_path, summary_path = save_promotion_report(
        _FakeVersion(), None, challenger_metrics, None, None, output_dir
    )

    assert csv_path.exists()
    assert summary_path.exists()
    assert "Bootstrap" in summary_path.read_text(encoding="utf-8")
    rows = pd.read_csv(csv_path)
    assert list(rows["role"]) == ["challenger"]


def test_save_promotion_report_with_decision_includes_both_rows(tmp_path: Path) -> None:
    X, y = _dataset()
    pipeline = _fit_pipeline(X, y)
    pipeline_path = tmp_path / "pipeline.pkl"
    save_pipeline_artifact(pipeline, pipeline_path)
    metrics, _ = evaluate_pipeline(
        load_local_pipeline(pipeline_path), X, y, REGISTRY_CONFIG["segment_columns"]
    )
    decision = decide_promotion(0.5, 0.6, {}, {}, REGISTRY_CONFIG)

    class _FakeVersion:
        def __init__(self, version: str) -> None:
            self.version = version

    output_dir = tmp_path / "reports"
    csv_path, summary_path = save_promotion_report(
        _FakeVersion("2"), _FakeVersion("1"), metrics, metrics, decision, output_dir
    )

    rows = pd.read_csv(csv_path)
    assert set(rows["role"]) == {"challenger", "champion"}
    assert "Promovido" in summary_path.read_text(encoding="utf-8")
