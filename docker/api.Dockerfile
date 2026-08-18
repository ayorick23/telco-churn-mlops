# Imagen de la API de predicción (Fase 8). Build reproducible: uv instala
# el grupo base de pyproject.toml (sin el grupo "dev" — ruff/mypy/pytest no
# pertenecen a producción). Precondición: `dvc pull models/champion_pipeline.pkl.dvc`
# debe haberse corrido ANTES de este build (ver .github/workflows/ci-cd.yml)
# — la imagen empaqueta el .pkl real, no lo descarga en runtime (ADR 0012:
# nunca depender de un servicio externo estar arriba para servir predicciones).
#
# Build local:
#   dvc pull models/champion_pipeline.pkl.dvc
#   docker build -f docker/api.Dockerfile -t churn-api .

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src/ src/
RUN uv sync --locked --no-dev

FROM python:3.11-slim

# libgomp1: runtime de OpenMP requerido por LightGBM (libgomp.so.1) — el
# builder lo trae vía apt-get transitivo de gcc, pero esta etapa final es
# python:3.11-slim "limpio" y no lo incluye.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src
COPY configs/ configs/
COPY models/champion_pipeline.pkl models/champion_pipeline.pkl

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "churn_mlops.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
