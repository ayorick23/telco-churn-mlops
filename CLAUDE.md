# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

End-to-end MLOps system for telecom customer churn prediction. The goal is not just a trained model, but a complete production lifecycle: experiment tracking, model registry with champion/challenger promotion, drift detection, and automatic retraining. Dataset: Telco Customer Churn (7 043 rows × 50 columns), tracked by DVC with remote on DagsHub.

## Development commands

All dependency management uses **uv** (not pip or poetry). Python version: 3.11 (managed by uv).

```bash
uv sync                    # install all dependencies (creates .venv with Python 3.11)
uv run pytest              # run all tests
uv run pytest tests/unit   # run a single test folder
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src/           # type-check (runs in CI, not in pre-commit — too slow)
uv run pre-commit install  # install git hooks (one-time, after cloning)
```

DVC:

```bash
dvc pull                   # download data after cloning (requires DagsHub credentials)
dvc repro                  # run the full pipeline (data → features → training)
dvc repro <stage>          # run a single named stage
dvc push                   # upload new data versions to DagsHub
```

Docker:

```bash
docker compose up          # start all services (MLflow, API, dashboard, monitoring)
docker compose up mlflow   # start a single service
```

## Architecture

Unidirectional layer dependency — each layer only imports from layers above it:

```plain text
Data → Features → Training → Registry/Serving → Monitoring → Presentation
```

```plain text
src/churn_mlops/
├── data/          # ingestion + Pandera schema validation + drift injection (batches)
├── features/      # feature engineering only — no model logic
├── training/      # training, evaluation, hyperparameter search (Optuna), MLflow logging
├── registry/      # champion/challenger promotion logic (MLflow Model Registry)
├── monitoring/    # Evidently AI drift detection + retraining trigger
├── serving/api/   # FastAPI app + Pydantic schemas + SHAP explanations
└── dashboard/     # Streamlit app
```

**Two validation boundaries** — Pandera validates raw batch DataFrames; Pydantic validates individual API request objects. Do not swap these.

**Champion/challenger**: the registry layer compares the newly trained challenger against the current production model on a held-out set. Promotion only if the challenger wins by a predefined margin, verified across segments — not just aggregate F1.

## Key conventions

- **Src layout**: all application code under `src/churn_mlops/`. Always install via `uv sync` before running; imports like `from churn_mlops.data import ...`.
- **Tests**: mirror `src/` structure under `tests/unit/`, `tests/integration/`, `tests/data_validation/`.
- **Data**: `data/` is DVC-managed. The `.dvc` pointer files go to git; actual data goes to DagsHub. Run `dvc pull` after cloning.
- **Notebooks**: `notebooks/` is exploration only. No production logic lives there.
- **Config**: environment-specific settings go in `configs/*.yaml`. No hardcoded paths or thresholds in source code.
- **MLflow**: all training runs log params, metrics, and artifacts to the DagsHub-hosted MLflow server. URI is set via `MLFLOW_TRACKING_URI` env var (see `.env.example`).
- **Commits**: Conventional Commits style (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- **Branches**: one `feat/<scope>` branch per phase/feature, one PR per branch, merged to `main`.
- **Design decisions**: significant architecture/tooling decisions are logged as ADRs in `docs/decisions/`. Check there before proposing an alternative to something already decided.
- **Learning goal**: this is an individual portfolio project — the point is to learn MLOps end-to-end, not just to ship a working pipeline. When a phase introduces a new library/tool (Optuna in Phase 4, Evidently AI in Phase 5), explain its core concepts, syntax, and the design decisions behind how it was used — don't just write the code and move on. At the end of each phase, do a detailed walkthrough of the new tools used in that phase, even if not explicitly asked.

## Infrastructure and credentials

| Service | URL | Purpose |
|---|---|---|
| DVC remote | `https://dagshub.com/ayorick23/telco-churn-mlops.dvc` | Data versioning |
| MLflow | `https://dagshub.com/ayorick23/telco-churn-mlops.mlflow` | Experiment tracking + Model Registry |
| FastAPI | Render / Railway (free tier) | REST prediction API |
| Streamlit | Hugging Face Spaces | Dashboard |
| GitHub Actions | — | CI (lint, type-check, tests) + CD |

Credentials go in `.env` (gitignored). Use `.env.example` as the template. DVC credentials also go in `.dvc/config.local` (gitignored).

## Known dependency quirk

`shap` has a transitive dependency on `llvmlite` via `numba`. Without explicit version bounds, uv resolves `llvmlite==0.36.0` which only supports Python <3.10. Fixed by pinning in `pyproject.toml`:

```toml
"llvmlite>=0.41",
"numba>=0.57",
```

Do not remove these lines.

## Project status (as of 2026-08-14)

| Phase | Status |
|---|---|
| 1. Setup y EDA | ✅ Completado |
| 2. Data layer + features | ✅ Completado |
| 3. Selección de modelo | ✅ Completado |
| 4. Training pipeline definitivo | ✅ Completado |
| 5. Simulación de drift + monitoreo | ✅ Completado |
| 6. Reentrenamiento y promoción | ✅ Completado |
| 7. Serving + dashboard | ⬜ Pendiente |
| 8. Testing, CI/CD, despliegue | ⬜ Pendiente |
