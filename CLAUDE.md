# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

End-to-end MLOps system for telecom customer churn prediction. The goal is not just a trained model, but a complete production lifecycle: experiment tracking, model registry with champion/challenger promotion, drift detection, and automatic retraining. Dataset: Telco Customer Churn (7 043 rows × 50 columns, in `data/raw/telco.csv`).

## Development commands

All dependency management uses **uv** (not pip or poetry).

```bash
uv sync                   # install all dependencies
uv run pytest             # run all tests
uv run pytest tests/unit  # run a single test folder
uv run ruff check .       # lint
uv run ruff format .      # format
uv run mypy src/          # type-check
```

DVC pipeline:
```bash
dvc repro                 # run the full pipeline (data → features → training)
dvc repro train           # run a single stage by name
```

Docker:
```bash
docker compose up         # start all services (MLflow, API, dashboard, monitoring)
docker compose up mlflow  # start a single service
```

## Architecture

Unidirectional layer dependency: **Data → Features → Training → Registry/Serving → Monitoring → Presentation**. Each layer only imports from layers above it — monitoring depends on serving and training, but neither depends on monitoring.

```
src/churn_mlops/
├── data/          # ingestion + Pandera schema validation + drift injection
├── features/      # feature engineering (no model logic here)
├── training/      # model training, hyperparameter search, evaluation
├── registry/      # champion/challenger promotion logic (MLflow Model Registry)
├── monitoring/    # Evidently AI drift detection + retraining trigger
├── serving/api/   # FastAPI app + Pydantic request/response schemas + SHAP
└── dashboard/     # Streamlit app
```

**Two validation boundaries** — Pandera validates raw batch data (CSV/DataFrames), Pydantic validates individual API requests. Do not swap these.

**Champion/challenger flow**: the registry layer compares the newly trained challenger against the current production champion on a held-out set; promotion happens only if the challenger wins by a predefined margin. A/B testing with live traffic is out of scope by design.

## Key conventions

- **Src layout**: all application code lives under `src/churn_mlops/`. Imports always use the full package path (`from churn_mlops.data import ...`).
- **Tests**: organized as `tests/unit/`, `tests/integration/`, `tests/data_validation/`. Integration tests may require MLflow running locally.
- **Data versioning**: `data/` is managed by DVC, not committed to git. Running `dvc pull` is required after cloning.
- **Notebooks**: `notebooks/` is for exploration only — no production logic should live there.
- **Configs**: environment-specific configs go in `configs/`. No hardcoded paths or thresholds in source code.
- **Experiment tracking**: all training runs must log params, metrics, and artifacts to MLflow. The MLflow tracking URI is set via environment variable (`MLFLOW_TRACKING_URI`).

## Infrastructure

| Service      | Purpose                                  |
|--------------|------------------------------------------|
| MLflow       | Experiment tracking + Model Registry     |
| FastAPI      | REST prediction API + SHAP explanations  |
| Streamlit    | Dashboard (deployed to Hugging Face Spaces) |
| Evidently AI | Drift detection reports                  |
| GitHub Actions | CI (lint, type-check, tests) + CD      |

CI runs Ruff, MyPy, and pytest on every PR. Pre-commit hooks enforce the same checks locally.
