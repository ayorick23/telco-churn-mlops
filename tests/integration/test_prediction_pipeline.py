"""Pipeline champion real (sin mocks) de punta a punta, vía TestClient.

tests/unit/serving/api/test_main.py mockea get_pipeline/get_champion_metadata/
explain_prediction a propósito — cubre el contrato HTTP, nunca el .pkl real
(ver su docstring: "no la corrección numérica del modelo/SHAP, ya verificada
manualmente"). Ese "verificada manualmente" es exactamente el hueco que
cierra este archivo: correr feature engineering + predict_proba + el
explainer de SHAP reales, sin mocks, contra el champion versionado.

Requiere `dvc pull models/champion_pipeline.pkl.dvc` antes de correr — se
salta con un mensaje claro si el archivo no está, para que `uv run pytest`
siga siendo seguro en una máquina que no hizo el pull."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from churn_mlops.serving.api import main as main_module

pytestmark = pytest.mark.skipif(
    not Path("models/champion_pipeline.pkl").exists(),
    reason="Falta models/champion_pipeline.pkl — correr "
    "`dvc pull models/champion_pipeline.pkl.dvc`",
)


@pytest.fixture
def client() -> TestClient:
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_predict_with_real_champion_pipeline(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    response = client.post("/predict", json=[valid_customer_payload])

    assert response.status_code == 200
    body = response.json()[0]
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    # No exigimos champion_version != "unknown": por diseño (ver
    # main.py::lifespan) la API sirve predicciones aunque el Registry de
    # MLflow no sea alcanzable, y en ese caso "unknown" es el valor
    # correcto, no un bug. Este test corrió en una red que bloquea el TLS
    # de DagsHub (ver ADR "Multi-machine setup" en memoria) y "unknown" es
    # exactamente el comportamiento esperado ahí.
    assert isinstance(body["champion_version"], str)


def test_explain_with_real_champion_pipeline_returns_shap_values(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    response = client.post("/explain", json=valid_customer_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert len(body["shap_values"]) > 0
    assert all(isinstance(v, float) for v in body["shap_values"].values())
