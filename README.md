# Churn MLOps — Sistema de Predicción de Churn con Ciclo Completo de MLOps

> 🚧 **Proyecto en desarrollo activo.** Este README se actualizará a medida que avancen las fases del roadmap.

Sistema end-to-end de predicción de churn de clientes que va más allá de entrenar un modelo: implementa el ciclo completo de MLOps — tracking de experimentos, registro y promoción de modelos, detección de drift, y reentrenamiento asistido — sobre un dataset de churn de telecomunicaciones (Telco Customer Churn).

El objetivo del proyecto no es solo predecir qué clientes van a cancelar su servicio, sino demostrar cómo se mantiene un modelo de ML confiable en producción a lo largo del tiempo, a medida que los datos y el comportamiento de los clientes cambian.

## Tabla de contenidos

- [Churn MLOps — Sistema de Predicción de Churn con Ciclo Completo de MLOps](#churn-mlops--sistema-de-predicción-de-churn-con-ciclo-completo-de-mlops)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [Problema de negocio](#problema-de-negocio)
  - [Qué hace el sistema](#qué-hace-el-sistema)
  - [Qué no hace (por diseño)](#qué-no-hace-por-diseño)
  - [Arquitectura](#arquitectura)
  - [Stack tecnológico](#stack-tecnológico)
  - [Estructura del repositorio](#estructura-del-repositorio)
  - [Estado del proyecto](#estado-del-proyecto)
  - [Cómo ejecutarlo](#cómo-ejecutarlo)
  - [Roadmap](#roadmap)
  - [Licencia](#licencia)

## Problema de negocio

Retener un cliente existente cuesta significativamente menos que adquirir uno nuevo. Este proyecto simula el sistema que un equipo de datos construiría para: (1) identificar clientes con alta probabilidad de cancelar, permitiendo intervención proactiva, y (2) garantizar que ese modelo predictivo siga siendo confiable con el tiempo, detectando automáticamente cuándo su desempeño se degrada y reentrenándolo cuando corresponde — sin intervención manual constante.

## Qué hace el sistema

- Entrena y compara distintas familias de modelos (Regresión Logística como baseline, XGBoost, LightGBM, CatBoost) con tracking completo de experimentos en MLflow.
- Registra el modelo ganador en un Model Registry y promueve versiones vía el alias `champion` (API de aliases de MLflow, no el modelo de stages Staging→Production).
- Sirve predicciones de churn vía una API REST (FastAPI), incluyendo explicabilidad por predicción individual (SHAP).
- Simula la llegada de nuevos datos con distintos tipos de _drift_ documentados (covariate shift, concept drift) y los detecta con Evidently AI.
- Evalúa si el drift detectado amerita reentrenar, compara el modelo nuevo (_challenger_) contra el actual (_champion_) con un criterio de promoción explícito, y solo lo reemplaza si es realmente mejor — reentrenar y promover son pasos deliberadamente manuales, no un orquestador automático (ver ADR 0012).
- Expone un dashboard (Streamlit) con predicciones interactivas y su explicación en SHAP, métricas del modelo en producción, historial de drift, y una sección de operaciones para disparar monitoreo/reentrenamiento/promoción desde la UI.

## Qué no hace (por diseño)

Decisiones deliberadas de alcance, no limitaciones por descuido:

- No sirve predicciones en streaming — el caso de uso no lo requiere.
- No implementa autenticación o multi-tenancy tipo SaaS.
- No hace A/B testing con tráfico real — la comparación champion/challenger se hace sobre datos held-out.
- No usa orquestadores como Airflow/Prefect — la escala del pipeline (4-5 stages) no lo justifica; se usa DVC + GitHub Actions.

## Arquitectura

El sistema está organizado en capas con dependencias en una sola dirección:

```plain text
Data → Features → Training → Model Registry / Serving → Monitoring
                                        ↓
                                  Presentation (API + Dashboard)
```

Cada capa solo conoce a la anterior — por ejemplo, el sistema de monitoreo depende del serving y del training, pero ninguno de esos dos depende del monitoreo, lo que permite que el sistema siga entrenando y sirviendo predicciones aunque el monitoreo esté temporalmente caído.

## Stack tecnológico

| Categoría                      | Herramienta                                              |
| ------------------------------ | -------------------------------------------------------- |
| Modelado                       | Scikit-Learn, XGBoost, LightGBM, CatBoost                |
| Tracking / Registry            | MLflow                                                   |
| Versionado de datos y pipeline | DVC                                                      |
| Validación de datos            | Pandera (datos crudos/lotes), Pydantic (requests de API) |
| Monitoreo de drift             | Evidently AI                                             |
| Explicabilidad                 | SHAP                                                     |
| API                            | FastAPI                                                  |
| Dashboard                      | Streamlit (desplegado en Hugging Face Spaces)            |
| Testing                        | Pytest                                                   |
| Calidad de código              | Ruff, MyPy, pre-commit                                   |
| CI/CD                          | GitHub Actions                                           |
| Empaquetado                    | uv                                                       |
| Contenedores                   | Docker, Docker Compose                                   |

## Estructura del repositorio

```plain text
churn-mlops/
├── src/churn_mlops/
│   ├── data/            # ingesta, validación (Pandera), inyección de drift
│   ├── features/        # feature engineering
│   ├── training/         # entrenamiento, selección de modelo, evaluación
│   ├── registry/         # lógica de promoción champion/challenger
│   ├── monitoring/       # detección de drift, disparo de reentrenamiento
│   ├── serving/api/      # FastAPI
│   └── dashboard/        # Streamlit
├── tests/                 # unit, integration, data_validation
├── configs/               # configuración por ambiente
├── data/                  # gestionado por DVC
├── models/                # pipelines serializados, gestionado por DVC
├── reports/               # resultados de selección de modelo, drift y promoción
├── notebooks/             # solo exploración
├── docs/decisions/        # ADRs — decisiones de arquitectura y tooling
├── docker/
├── .github/workflows/
├── docker-compose.yml
├── pyproject.toml
├── dvc.yaml
├── LICENSE
└── README.md
```

## Estado del proyecto

| Fase                                         | Estado        |
| -------------------------------------------- | ------------- |
| 1. Setup y EDA                               | ✅ Completado |
| 2. Data layer + features                     | ✅ Completado |
| 3. Selección de modelo                       | ✅ Completado |
| 4. Training pipeline definitivo              | ✅ Completado |
| 5. Simulación de drift + monitoreo           | ✅ Completado |
| 6. Reentrenamiento y promoción               | ✅ Completado |
| 7. Serving + dashboard                       | ✅ Completado |
| 8. Testing, CI/CD, despliegue, documentación | ⬜ Pendiente  |

## Cómo ejecutarlo

Requiere Python 3.11 (gestionado por [uv](https://docs.astral.sh/uv/)) y
credenciales de DagsHub para `dvc pull`/MLflow (ver `.env.example`).

```bash
# clonar el repo
git clone https://github.com/ayorick23/telco-churn-mlops.git
cd telco-churn-mlops

# instalar dependencias (crea .venv con Python 3.11)
uv sync

# copiar el template de variables de entorno y completar credenciales de DagsHub
cp .env.example .env

# descargar datos y modelos versionados con DVC
dvc pull
```

### Pipeline de datos y entrenamiento

```bash
# valida el CSV crudo contra el schema de Pandera, genera train/test.parquet
# y entrena+tunea el pipeline de LightGBM (Optuna)
dvc repro
```

### Servir el modelo

```bash
# API de predicción (FastAPI + SHAP) — docs interactivas en /docs
uv run uvicorn churn_mlops.serving.api.main:app --reload --port 8000

# en otra terminal: dashboard (Streamlit)
uv run streamlit run src/churn_mlops/dashboard/app.py
```

### Monitoreo y reentrenamiento (manual — ver ADR 0012)

```bash
uv run python -m churn_mlops.monitoring.run_monitoring       # simula drift y lo detecta con Evidently
uv run python -m churn_mlops.monitoring.run_retrain_check    # ¿el drift amerita reentrenar?
uv run python -m churn_mlops.training.run_retrain             # reentrena el challenger
uv run python -m churn_mlops.registry.run_promotion            # compara y promueve a champion si corresponde
```

Los mismos pasos de monitoreo/reentrenamiento/promoción también se pueden
disparar desde la pestaña "Operaciones" del dashboard.

### Tests y calidad de código

```bash
uv run pytest              # suite completa
uv run ruff check .        # lint
uv run mypy src/           # type-check
```

> Docker y despliegue (Render/Railway + Hugging Face Spaces) quedan para la
> Fase 8 — `docker-compose.yml` está reservado pero todavía vacío.

## Roadmap

El roadmap del proyecto son las 8 fases de la tabla de "Estado del proyecto". Las decisiones de diseño y tooling tomadas en cada fase quedan registradas como ADRs en [`docs/decisions/`](docs/decisions/). El EDA completo de la Fase 1 está en [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Licencia

Distribuido bajo licencia [MIT](LICENSE).

---

_Proyecto de portafolio personal — no reutiliza código ni datos de ningún proyecto profesional o propiedad de terceros._
