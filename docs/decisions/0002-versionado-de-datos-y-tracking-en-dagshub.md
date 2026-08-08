# 2. Versionado de datos y tracking de experimentos en DagsHub

**Fecha:** 2026-08-07
**Estado:** Aceptada
**Fase:** 1 — Setup y EDA

## Contexto

El dataset (7 043 filas × 50 columnas) necesita versionado propio, y el proyecto necesita tracking de experimentos + un Model Registry para el flujo champion/challenger de fases posteriores. Al ser un proyecto de portafolio individual, la solución debe ser gratuita y no requerir mantener infraestructura propia.

## Decisión

- **DVC** para versionado de datos, con remoto en **DagsHub** (`https://dagshub.com/ayorick23/telco-churn-mlops.dvc`).
- **MLflow** con tracking server hospedado también en DagsHub, mismo proyecto (`https://dagshub.com/ayorick23/telco-churn-mlops.mlflow`).

## Alternativas consideradas

- **Git LFS** — versiona archivos grandes pero no da un pipeline de datos (`dvc.yaml`) ni se integra con MLflow.
- **S3/GCS como remoto de DVC** — requiere cuenta cloud de pago y credenciales adicionales para un proyecto sin presupuesto.
- **MLflow local o self-hosted** — mantenimiento extra, no accesible como evidencia de portafolio sin exponer un servidor propio.

## Consecuencias

- Un solo proveedor (DagsHub) centraliza datos, experimentos y registry, gratis para repos públicos.
- Credenciales van en `.env` (tracking URI, tokens) y `.dvc/config.local` (ambos gitignored).
- `MLFLOW_TRACKING_URI` se inyecta por variable de entorno, nunca hardcodeado en `src/`.
