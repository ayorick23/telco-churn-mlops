"""Tests de la API vía TestClient. Mockean get_pipeline/get_champion_metadata/
explain_prediction (nunca cargan el .pkl real ni pegan contra MLflow/DagsHub)
— cubren el contrato HTTP, no la corrección numérica del modelo/SHAP (ya
verificada manualmente contra el champion real, ver ADR 0013)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from churn_mlops.serving.api import main as main_module
from fastapi.testclient import TestClient


class _FakePipeline:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.7, 0.3], (len(X), 1))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main_module, "get_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        main_module,
        "get_champion_metadata",
        lambda: ("churn-lightgbm", "champion", "3"),
    )
    monkeypatch.setattr(
        main_module,
        "explain_prediction",
        lambda pipeline, X: ({"numeric__Age": 0.12}, -1.5),
    )
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_returns_champion_metadata(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["champion_version"] == "3"
    assert body["registered_model_name"] == "churn-lightgbm"


def test_predict_returns_one_prediction_per_customer(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    response = client.post("/predict", json=[valid_customer_payload])

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["churn_probability"] == pytest.approx(0.3)
    assert body[0]["champion_version"] == "3"


def test_predict_rejects_invalid_categorical(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    payload = {**valid_customer_payload, "Contract": "Not-A-Real-Contract"}

    response = client.post("/predict", json=[payload])

    assert response.status_code == 422


def test_explain_returns_shap_values(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    response = client.post("/explain", json=valid_customer_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["shap_values"] == {"numeric__Age": 0.12}
    assert body["base_value"] == -1.5
