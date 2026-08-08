# 1. Gestión de dependencias y entorno

**Fecha:** 2026-08-07
**Estado:** Aceptada
**Fase:** 1 — Setup y EDA

## Contexto

El proyecto tiene dependencias pesadas de ML (scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, MLflow, DVC). Se necesita un gestor de paquetes rápido, con lockfile determinista, y un build backend simple ya que el paquete no se publica a PyPI — solo se instala localmente vía `src/` layout.

## Decisión

- **uv** como gestor de dependencias y entornos virtuales (no pip+venv manual, no Poetry).
- **hatchling** como build backend (no setuptools).
- **Python 3.11** fijado y gestionado por uv (no la versión de sistema).

## Alternativas consideradas

- **Poetry** — resolución de dependencias más lenta, tooling adicional que uv ya cubre.
- **pip + requirements.txt** — sin lockfile determinista, instalación reproducible más frágil.
- **conda** — overkill para este proyecto; no hay dependencias no-Python que lo justifiquen.

## Consecuencias

- Todos los comandos del proyecto se ejecutan con `uv run`.
- `uv.lock` se versiona en git para reproducibilidad exacta del entorno.
- Onboarding de un colaborador nuevo se reduce a `uv sync`.
