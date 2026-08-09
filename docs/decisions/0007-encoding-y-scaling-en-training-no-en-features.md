# 7. Encoding y scaling viven en training/, no en features/

**Fecha:** 2026-08-08
**Estado:** Aceptada
**Fase:** 2 — Data layer + features

## Contexto

`Churn Label` se predice con al menos dos familias de modelos (Regresión Logística como baseline, árboles como XGBoost/LightGBM/CatBoost en fases posteriores). La Regresión Logística necesita categóricas codificadas (one-hot) y numéricas escaladas; los modelos de árboles no lo necesitan y en algunos casos (CatBoost, LightGBM) manejan categóricas nativamente mejor que con one-hot. Había que decidir en qué capa vive ese encoding/scaling.

## Decisión

`src/churn_mlops/features/build_features.py` (Fase 2) entrega un DataFrame limpio, tipado y con las columnas derivadas, pero **las categóricas quedan como texto** (`Contract`, `Payment Method`, etc., sin one-hot). El `ColumnTransformer`/`Pipeline` de sklearn que hace el encoding y el scaling se define en `training/` (Fase 3) y se fitea **solo sobre el split de train**.

## Alternativas consideradas

- **Encoding en `features/`** — features/ entregaría una matriz numérica lista para cualquier modelo. Rechazada: el encoder se fitearía sobre el dataset completo antes del split, filtrando información de test hacia train (leakage); además mezclaría una decisión específica de modelo dentro de una capa que `CLAUDE.md` define como "feature engineering only — no model logic".

## Consecuencias

- Cada familia de modelo en Fase 3 arma su propio preprocesamiento (p. ej. `ColumnTransformer` con OneHotEncoder + StandardScaler para logística; passthrough para árboles) dentro de un `Pipeline` de sklearn, fiteado únicamente sobre `X_train`.
- `features/` no necesita cambiar cuando se agregue o quite una familia de modelos.
- `train.parquet`/`test.parquet` (ver [0008](0008-split-train-test-versionado-en-fase-2.md)) tienen categóricas como string — quien los consuma en Fase 3 debe encodearlas antes de entrenar.
