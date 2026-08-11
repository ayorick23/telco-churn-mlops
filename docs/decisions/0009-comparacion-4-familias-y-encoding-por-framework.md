# 9. Comparación de 4 familias de modelo con encoding específico por framework

**Fecha:** 2026-08-11
**Estado:** Aceptada
**Fase:** 3 — Selección de modelo

## Contexto

Fase 3 compara 4 familias de modelo (Regresión Logística como baseline, XGBoost,
LightGBM, CatBoost) sobre el mismo `train.parquet`/`test.parquet` (ADR 0008). Es una
comparación deliberada para practicar los tres frameworks de gradient boosting, no
solo para elegir un ganador. `City` tiene 1104 valores únicos en train y no fue
excluida en Fase 2 (solo se excluyeron Latitude/Longitude/Zip Code/CLTV como
low-signal) — one-hot encodearla generaría ~1104 columnas. Había que decidir cómo
encodean categóricas las 4 familias, cómo tratan `City`, cuándo entra Optuna, y si
Fase 3 se integra a `dvc.yaml`.

## Decisión

- Las 4 familias se entrenan con hiperparámetros por defecto o manualmente
  razonables (sin Optuna) y se loguean como runs separados en el experimento MLflow
  `phase3-model-selection`. La familia ganadora se elige por F1 de la clase positiva
  (churn=1) sobre `test.parquet`; accuracy, precision, recall, ROC-AUC y PR-AUC se
  registran como métricas secundarias.
- Cada framework encodea categóricas a su manera (ADR 0007): Regresión Logística,
  XGBoost y LightGBM arman un `Pipeline` de sklearn (`ColumnTransformer` con
  `SimpleImputer` + `OneHotEncoder`, más `StandardScaler` solo para Regresión
  Logística) fiteado únicamente sobre `X_train`. CatBoost usa sus categóricas
  nativas vía `cat_features`, sin one-hot, manejando NaN nativamente en numéricas
  (las categóricas sí necesitan un placeholder de texto para sus NaN estructurales).
- `City` se dropea (vía `remainder='drop'` del `ColumnTransformer`) para Regresión
  Logística, XGBoost y LightGBM — `Population` ya sirve de proxy numérico del
  tamaño/tipo de zona geográfica, igual criterio que llevó a dropear
  Latitude/Longitude/Zip Code en Fase 2. Para CatBoost, `City` se pasa nativa como
  categórica: es el caso de uso que justifica elegir CatBoost para alta
  cardinalidad.
- Optuna se reserva para Fase 4, aplicado únicamente a la familia ganadora de Fase
  3 (ya anotado en `pyproject.toml`).
- Fase 3 se corre a mano (`uv run python -m churn_mlops.training.run_model_selection`),
  no es un stage de `dvc.yaml`. Fase 4 integra a `dvc.yaml` solo el pipeline de
  entrenamiento de la familia ganadora.

## Alternativas consideradas

- **One-hot con `handle_unknown="infrequent_if_exist"` + `min_frequency` para
  agrupar ciudades poco frecuentes**, en vez de dropear `City` en Regresión
  Logística/XGBoost/LightGBM. Rechazada para esta fase: agrega un hiperparámetro
  más (`min_frequency`) a tunear en una comparación pensada para tener esfuerzo de
  tuning equivalente entre las 4 familias; `Population` ya es un proxy razonable.
  Si la familia ganadora en Fase 4 resulta ser una de estas tres y un análisis de
  importancia de features muestra que `City` aporta señal relevante, se puede
  reconsiderar como parte del tuning con Optuna.
- **Optuna desde Fase 3 sobre las 4 familias** — rechazada por costo: tunear 4
  familias completas antes de elegir ganadora multiplica el cómputo sin cambiar la
  decisión de selección en la mayoría de los casos (el ranking relativo con
  hiperparámetros default suele mantenerse). Se prefiere gastar ese presupuesto de
  búsqueda solo en la familia ganadora.
- **Integrar el entrenamiento de Fase 3 como stage de `dvc.yaml`** — rechazada: DVC
  versiona pipelines deterministas de producción; con 4 familias en modo
  exploratorio, forzar un stage reproducible agrega fricción sin beneficio hasta
  saber cuál familia se queda.

## Consecuencias

- `training/model_specs.py` concentra la única lógica que sabe qué `Pipeline` arma
  cada framework; agregar una quinta familia no toca `features/` ni `data/`.
- Si `City` resulta relevante para la familia ganadora, reconsiderar su encoding es
  un ítem de Fase 4, no de Fase 3.
- El experimento MLflow `phase3-model-selection` queda como el registro histórico de
  por qué se eligió la familia ganadora; `reports/model_selection_summary.md` es la
  versión legible del mismo veredicto, versionada en el repo.
