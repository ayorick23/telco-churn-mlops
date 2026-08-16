"""Carga el Pipeline de producción para la API. Reusa
`registry/run_promotion.py::load_local_pipeline` (mismo joblib.load ya usado
para leer challenger/champion en Fase 6) en vez de duplicarlo.

Dos responsabilidades separadas a propósito:
- `get_pipeline()`: solo necesita el archivo local (ADR 0012) — /predict y
  /explain no dependen de que el servidor de MLflow esté arriba.
- `get_champion_metadata()`: sí pega contra el Registry de MLflow (solo
  metadata, nunca bytes) — únicamente la usa /model-info, que puede fallar
  sin afectar al resto de la API.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline

from churn_mlops.config import load_yaml_config
from churn_mlops.registry.mlflow_client import get_champion_version
from churn_mlops.registry.run_promotion import load_local_pipeline

_SERVING_CONFIG_PATH = "configs/serving.yaml"


@lru_cache(maxsize=1)
def get_serving_config() -> dict[str, Any]:
    return load_yaml_config(_SERVING_CONFIG_PATH)


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """Carga models/champion_pipeline.pkl una sola vez por proceso. Falla
    rápido y claro si no está — nunca cae a models/lightgbm_pipeline.pkl
    (challenger no promovido, violaría la garantía de ADR 0012 de que el
    champion es la única fuente de verdad de producción)."""
    model_path = Path(get_serving_config()["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró {model_path}. Correr `dvc pull` para traer el "
            "champion versionado, o `uv run python -m "
            "churn_mlops.registry.run_promotion` si todavía no hay ningún "
            "champion promovido."
        )
    return load_local_pipeline(model_path)


def get_champion_metadata() -> tuple[str, str, str | None]:
    """(registered_model_name, champion_alias, champion_version). La versión
    es None si el Registry no es alcanzable o si todavía no hay champion
    (caso bootstrap, ver ADR 0012) — el caller decide cómo comunicarlo."""
    serving_config = get_serving_config()
    model_name = serving_config["mlflow"]["registered_model_name"]
    champion_alias = serving_config["mlflow"]["champion_alias"]

    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.MlflowClient()
    champion_version = get_champion_version(client, model_name, champion_alias)
    version = champion_version.version if champion_version is not None else None
    return model_name, champion_alias, version
