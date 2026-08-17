# Imagen del dashboard (Fase 8). El dashboard es un cliente HTTP puro de la
# API (dashboard/api_client.py) — nunca carga el .pkl del modelo ni necesita
# credenciales de DagsHub, a diferencia de docker/api.Dockerfile. Apunta a la
# API vía la env var CHURN_API_BASE_URL (configs/dashboard.yaml:api_base_url
# es el default de desarrollo local).
#
# Build local:
#   docker build -f docker/dashboard.Dockerfile -t churn-dashboard .

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

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src
COPY configs/ configs/
COPY reports/ reports/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "src/churn_mlops/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
