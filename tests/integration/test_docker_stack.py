"""El stack completo de docker-compose.yml (API + dashboard), construido y
levantado de verdad — nunca antes ejercitado por automatización. El job `ci`
solo construía las imágenes (`docker build`, sin correrlas); el bug real que
esto habría atrapado (commit c006a49, 2026-08-17: faltaba libgomp1 en la
etapa runtime de docker/api.Dockerfile, crash loop al importar
churn_mlops.training.model_specs) solo se detectó al levantar el stack a
mano, porque un build exitoso no implica que el contenedor corra.

Se salta si no hay Docker disponible o si falta el .pkl del champion (la
imagen de la API no puede construirse sin él) — `uv run pytest` sigue
siendo seguro en una máquina sin Docker Desktop corriendo.

`docker-compose.yml` espera un archivo `.env` (env_file de `api`) que en un
checkout limpio de CI no existe (está en .gitignore) — si falta, este test
escribe uno mínimo y lo borra al terminar; si ya existe (dev local con
credenciales reales), lo deja intacto. /health y /predict no necesitan
MLflow (ver model_loader.py), así que valores vacíos alcanzan."""

import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest


def _docker_daemon_reachable() -> bool:
    """shutil.which("docker") solo confirma que el CLI está en el PATH, no
    que el daemon esté corriendo (ej. Docker Desktop cerrado) — `docker info`
    falla rápido en ese caso, así el skip es limpio en vez de un error feo de
    `docker compose up`."""
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode
        == 0
    )


_MODEL_PRESENT = Path("models/champion_pipeline.pkl").exists()

pytestmark = pytest.mark.skipif(
    not (_MODEL_PRESENT and _docker_daemon_reachable()),
    reason="Requiere el daemon de Docker corriendo y "
    "models/champion_pipeline.pkl (`dvc pull models/champion_pipeline.pkl.dvc`)",
)

_API_URL = "http://localhost:8000"
_DASHBOARD_URL = "http://localhost:8501"
_ENV_FILE = Path(".env")


def _wait_until_ok(url: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | str | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"status {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"{url} no respondió 200 en {timeout}s (último: {last_error})")


@pytest.fixture(scope="module")
def docker_stack() -> Iterator[None]:
    created_env_file = False
    if not _ENV_FILE.exists():
        _ENV_FILE.write_text(
            "MLFLOW_TRACKING_URI=\n"
            "MLFLOW_TRACKING_USERNAME=\n"
            "MLFLOW_TRACKING_PASSWORD=\n"
            "ENV=test\n",
            encoding="utf-8",
        )
        created_env_file = True

    try:
        # capture_output=True a propósito: el progreso "en vivo" de
        # BuildKit reescribe la terminal con secuencias ANSI, así que si
        # este build falla, dejarlo heredar stdout/stderr del proceso padre
        # no deja rastro legible en el log de CI. Capturado, el mensaje de
        # error de pytest incluye el log completo del build.
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "--build"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "`docker compose up --build` falló "
                f"(exit {result.returncode}):\n--- stdout ---\n{result.stdout}"
                f"\n--- stderr ---\n{result.stderr}"
            )
        _wait_until_ok(f"{_API_URL}/health")
        _wait_until_ok(f"{_DASHBOARD_URL}/_stcore/health")
        yield
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], check=False)
        if created_env_file:
            _ENV_FILE.unlink(missing_ok=True)


def test_api_health(docker_stack: None) -> None:
    response = httpx.get(f"{_API_URL}/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_predicts_through_the_real_container(
    docker_stack: None, valid_customer_payload: dict[str, Any]
) -> None:
    response = httpx.post(
        f"{_API_URL}/predict", json=[valid_customer_payload], timeout=10.0
    )

    assert response.status_code == 200
    body = response.json()[0]
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_dashboard_is_reachable_through_the_real_container(
    docker_stack: None,
) -> None:
    response = httpx.get(_DASHBOARD_URL, timeout=10.0)

    assert response.status_code == 200
