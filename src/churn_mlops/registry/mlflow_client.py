"""Wrapper delgado sobre el Model Registry de MLflow — API de ALIASES
(`set_registered_model_alias`/`get_model_version_by_alias`), no la API de
stages (`transition_model_version_stage`, deprecada desde MLflow 2.9 — ADR
0012). Aísla al resto de registry/ del cliente concreto, mismo rol que
drift_detection.py respecto a Evidently en Fase 5.

El Model Registry solo versiona el `LGBMClassifier` (run_training.py loguea
el preprocesador aparte, artifact_path="preprocessor", en el mismo run — ver
ADR 0012). Por eso evaluar cualquier `ModelVersion` requiere ir a su
`run_id` de origen y reconstruir el Pipeline completo combinando ambos
artifacts — ver load_pipeline_from_run."""

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from sklearn.pipeline import Pipeline


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


def load_pipeline_from_run(run_id: str) -> Pipeline:
    """Reconstruye el Pipeline completo (preprocesador + modelo) desde los dos
    artifacts logueados en el run de origen — el Registry solo versiona el
    segundo."""
    preprocessor = mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def promote_challenger(
    client: mlflow.MlflowClient, model_name: str, champion_alias: str, version: str
) -> None:
    """Mueve el alias de champion a `version`. MLflow permite un solo model
    version por alias — no hace falta despromover el champion anterior antes,
    sirve tanto para el bootstrap como para reemplazarlo."""
    client.set_registered_model_alias(model_name, champion_alias, version)
