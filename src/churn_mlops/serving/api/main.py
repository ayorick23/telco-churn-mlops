"""API de predicción de churn (Fase 7). Levanta con:

    uv run uvicorn churn_mlops.serving.api.main:app --reload

`GET /health` y `GET /model-info` son de diagnóstico; `POST /predict` y
`POST /explain` son los endpoints de negocio. `/predict`/`/explain` nunca
dependen de que el servidor de MLflow esté arriba (ADR 0012: el Pipeline se
lee de disco) — la versión del champion que devuelven se resuelve una sola
vez al arrancar el proceso, no en cada request; si el Registry no está
disponible en ese momento, siguen sirviendo predicciones con
`champion_version="unknown"`. `/model-info` sí pega contra el Registry en
cada llamada, porque su propósito es mostrar el estado vivo."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from churn_mlops.serving.api.explain import explain_prediction
from churn_mlops.serving.api.features import customers_to_model_input
from churn_mlops.serving.api.model_loader import (
    get_champion_metadata,
    get_pipeline,
    get_serving_config,
)
from churn_mlops.serving.api.schemas import (
    CustomerFeatures,
    ExplanationResponse,
    ModelInfoResponse,
    PredictionResponse,
)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Falla rápido y ruidoso si falta el .pkl (ver model_loader.get_pipeline) —
    # un proceso que no puede predecir no debería terminar de arrancar.
    get_pipeline()
    try:
        _model_name, _alias, champion_version = get_champion_metadata()
    except Exception:
        champion_version = None
    _state["champion_version"] = champion_version or "unknown"
    yield
    _state.clear()


serving_config = get_serving_config()
app = FastAPI(title=serving_config["api"]["title"], lifespan=lifespan)

# API y dashboard se despliegan en hosts distintos (Render/Railway vs
# Hugging Face Spaces, ver CLAUDE.md) — el dashboard pega cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        model_name, champion_alias, champion_version = get_champion_metadata()
    except (mlflow.exceptions.MlflowException, KeyError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Model Registry no disponible: {exc}"
        ) from exc
    return ModelInfoResponse(
        registered_model_name=model_name,
        champion_alias=champion_alias,
        champion_version=champion_version,
    )


@app.post("/predict", response_model=list[PredictionResponse])
def predict(customers: list[CustomerFeatures]) -> list[PredictionResponse]:
    pipeline = get_pipeline()
    X = customers_to_model_input(customers)
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = pipeline.predict(X)
    champion_version = _state["champion_version"]
    return [
        PredictionResponse(
            churn_probability=float(proba),
            churn_prediction=int(pred),
            champion_version=champion_version,
        )
        for proba, pred in zip(probabilities, predictions, strict=True)
    ]


@app.post("/explain", response_model=ExplanationResponse)
def explain(customer: CustomerFeatures) -> ExplanationResponse:
    pipeline = get_pipeline()
    X = customers_to_model_input([customer])
    proba = float(pipeline.predict_proba(X)[:, 1][0])
    pred = int(pipeline.predict(X)[0])
    shap_values, base_value = explain_prediction(pipeline, X)
    return ExplanationResponse(
        churn_probability=proba,
        churn_prediction=pred,
        champion_version=_state["champion_version"],
        shap_values=shap_values,
        base_value=base_value,
    )
