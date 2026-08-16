"""Cliente HTTP delgado hacia la API de Fase 7 (`serving/api/`). El dashboard
y la API se despliegan en hosts distintos (Render/Railway vs Hugging Face
Spaces, ver CLAUDE.md), por eso el dashboard le pega por HTTP en vez de
importar la lógica de predicción directamente."""

from typing import Any

import requests


def predict(api_base_url: str, customer_payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url}/predict", json=[customer_payload], timeout=10
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()[0]
    return result


def explain(api_base_url: str, customer_payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url}/explain", json=customer_payload, timeout=30
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def model_info(api_base_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_base_url}/model-info", timeout=10)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result
