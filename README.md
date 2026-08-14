# Churn MLOps — Sistema de Predicción de Churn con Ciclo Completo de MLOps

> 🚧 **Proyecto en desarrollo activo.** Este README se actualizará a medida que avancen las fases del roadmap.

Sistema end-to-end de predicción de churn de clientes que va más allá de entrenar un modelo: implementa el ciclo completo de MLOps — tracking de experimentos, registro y promoción de modelos, detección de drift, y reentrenamiento automático — sobre un dataset de churn de telecomunicaciones (Telco Customer Churn).

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

## Problema de negocio

Retener un cliente existente cuesta significativamente menos que adquirir uno nuevo. Este proyecto simula el sistema que un equipo de datos construiría para: (1) identificar clientes con alta probabilidad de cancelar, permitiendo intervención proactiva, y (2) garantizar que ese modelo predictivo siga siendo confiable con el tiempo, detectando automáticamente cuándo su desempeño se degrada y reentrenándolo cuando corresponde — sin intervención manual constante.

## Qué hace el sistema

- Entrena y compara distintas familias de modelos (Regresión Logística como baseline, XGBoost, LightGBM, CatBoost) con tracking completo de experimentos en MLflow.
- Registra el modelo ganador en un Model Registry con ciclo de vida de estados (Staging → Production).
- Sirve predicciones de churn vía una API REST (FastAPI), incluyendo explicabilidad por predicción individual (SHAP).
- Simula la llegada de nuevos datos con distintos tipos de _drift_ documentados (covariate shift, concept drift) y los detecta automáticamente con Evidently AI.
- Dispara reentrenamiento automático ante drift, compara el modelo nuevo (_challenger_) contra el actual (_champion_) con un criterio de promoción explícito, y solo lo reemplaza si es realmente mejor.
- Expone un dashboard (Streamlit) con predicciones, métricas del modelo en producción, alertas de drift e historial de reentrenamientos.

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
├── notebooks/             # solo exploración
├── docker/
├── .github/workflows/
├── docker-compose.yml
├── pyproject.toml
├── dvc.yaml
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
| 6. Reentrenamiento y promoción               | ⬜ Pendiente  |
| 7. Serving + dashboard                       | ⬜ Pendiente  |
| 8. Testing, CI/CD, despliegue, documentación | ⬜ Pendiente  |

## Cómo ejecutarlo

> Sección provisional — se completará a medida que el proyecto avance.

```bash
# clonar el repo
git clone <url-del-repo>
cd churn-mlops

# instalar dependencias con uv
uv sync

# descargar los datos versionados con DVC (requiere credenciales de DagsHub)
dvc pull

# correr el pipeline: valida el CSV crudo contra el schema de Pandera y genera
# data/processed/train.parquet y test.parquet
dvc repro

# levantar el entorno completo (entrenamiento, MLflow, monitoreo)
docker compose up
```

## Roadmap

El roadmap del proyecto son las 8 fases de la tabla de "Estado del proyecto". Las decisiones de diseño y tooling tomadas en cada fase quedan registradas como ADRs en [`docs/decisions/`](docs/decisions/). El EDA completo de la Fase 1 está en [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

---

_Proyecto de portafolio personal — no reutiliza código ni datos de ningún proyecto profesional o propiedad de terceros._
