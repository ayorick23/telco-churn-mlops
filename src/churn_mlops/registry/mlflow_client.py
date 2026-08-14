"""Wrapper delgado sobre el Model Registry de MLflow — API de ALIASES
(`set_registered_model_alias`/`get_model_version_by_alias`), no la API de
stages (`transition_model_version_stage`, deprecada desde MLflow 2.9 — ADR
0012). Aísla al resto de registry/ del cliente concreto, mismo rol que
drift_detection.py respecto a Evidently en Fase 5.

Solo maneja METADATA del Registry (qué versión existe, cuál es el alias
`champion`) — nunca descarga artifacts desde acá. Los bytes del pipeline
(challenger/champion) se resuelven en run_promotion.py desde archivos
locales versionados por DVC, no desde el Registry (ADR 0012: el proxy de
artifacts de MLflow en DagsHub resultó no ser confiable para descargas)."""

import mlflow
from mlflow.entities.model_registry import ModelVersion


def get_champion_version(
    client: mlflow.MlflowClient, model_name: str, champion_alias: str
) -> ModelVersion | None:
    """None si el alias todavía no existe — caso bootstrap explícito (ADR
    0012): la primera vez que corre run_promotion.py no hay ningún champion
    todavía."""
    try:
        return client.get_model_version_by_alias(model_name, champion_alias)
    except mlflow.exceptions.MlflowException:
        return None


def get_latest_version(client: mlflow.MlflowClient, model_name: str) -> ModelVersion:
    """El challenger es siempre la versión con el número más alto registrada
    (la que acaba de dejar run_training.py/run_retrain.py). No usa
    `get_latest_versions` (API de stages, deprecada)."""
    versions = client.search_model_versions(f"name='{model_name}'")
    return max(versions, key=lambda v: int(v.version))


def promote_challenger(
    client: mlflow.MlflowClient, model_name: str, champion_alias: str, version: str
) -> None:
    """Mueve el alias de champion a `version`. MLflow permite un solo model
    version por alias — no hace falta despromover el champion anterior antes,
    sirve tanto para el bootstrap como para reemplazarlo."""
    client.set_registered_model_alias(model_name, champion_alias, version)
