from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from churn_mlops.registry.mlflow_client import (
    get_champion_version,
    get_latest_version,
    promote_challenger,
)
from churn_mlops.training.model_specs import build_lightgbm_pipeline

MODEL_NAME = "test-churn-lightgbm-registry"


def _sample_data(n: int = 20) -> tuple[pd.DataFrame, pd.Series]:
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
    return X, y


def _log_and_register_version(model_name: str) -> str:
    """Loguea un run pequeño con preprocesador+modelo reales (mismo patrón que
    run_training.py::evaluate_and_log_final_model) y devuelve el version
    number registrado."""
    X, y = _sample_data()
    pipeline = build_lightgbm_pipeline(
        X, [], {"n_estimators": 5, "random_state": 42, "verbose": -1}
    )
    pipeline.fit(X, y)

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
    mlflow.set_experiment("test-phase6-registry")
    return mlflow.MlflowClient()


def test_get_champion_version_returns_none_before_alias_is_set(
    mlflow_client: mlflow.MlflowClient,
) -> None:
    _log_and_register_version(MODEL_NAME)

    result = get_champion_version(mlflow_client, MODEL_NAME, "champion")

    assert result is None


def test_get_champion_version_returns_version_after_alias_is_set(
    mlflow_client: mlflow.MlflowClient,
) -> None:
    version = _log_and_register_version(MODEL_NAME)
    mlflow_client.set_registered_model_alias(MODEL_NAME, "champion", version)

    result = get_champion_version(mlflow_client, MODEL_NAME, "champion")

    assert result is not None
    assert result.version == version


def test_get_latest_version_returns_highest_version_number(
    mlflow_client: mlflow.MlflowClient,
) -> None:
    _log_and_register_version(MODEL_NAME)
    second_version = _log_and_register_version(MODEL_NAME)

    result = get_latest_version(mlflow_client, MODEL_NAME)

    assert result.version == second_version


def test_promote_challenger_moves_alias_to_the_given_version(
    mlflow_client: mlflow.MlflowClient,
) -> None:
    first_version = _log_and_register_version(MODEL_NAME)
    second_version = _log_and_register_version(MODEL_NAME)
    promote_challenger(mlflow_client, MODEL_NAME, "champion", first_version)

    promote_challenger(mlflow_client, MODEL_NAME, "champion", second_version)

    champion = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert champion.version == second_version
