# 8. El split train/test se genera y versiona en Fase 2

**Fecha:** 2026-08-08
**Estado:** Aceptada
**Fase:** 2 — Data layer + features

## Contexto

Fase 3 va a comparar varias familias de modelos y, dentro de cada una, varias configuraciones de hiperparámetros — cada corrida se registra en MLflow. Para que esas comparaciones sean válidas, todas necesitan evaluarse sobre exactamente el mismo held-out set. Había que decidir si ese split lo genera Fase 2 (`features/`) como artefacto versionado, o si cada script de entrenamiento en Fase 3 lo hace en el momento.

## Decisión

`build_features.py` (Fase 2) hace el split train/test **estratificado**, con `test_size` y `random_state` fijos desde `configs/data.yaml`, y lo guarda como `data/processed/train.parquet` / `test.parquet`, versionado por el stage `build_features` de `dvc.yaml`.

## Alternativas consideradas

- **Split dentro de cada script de Fase 3** — funciona igual de bien si el `random_state` es constante en todos los scripts, pero el split queda implícito en código de entrenamiento duplicado en vez de ser un artefacto explícito y versionado por DVC. Rechazada por ser más frágil: basta con que un experimento futuro olvide fijar el seed para invalidar la comparación entre corridas de MLflow.

## Consecuencias

- Todo entrenamiento de Fase 3 en adelante lee `data/processed/train.parquet` y `test.parquet` directamente — no vuelve a splitear.
- Si el criterio de split cambia (otro `test_size`, otra semilla, K-fold en vez de holdout), el cambio se hace en un solo lugar (`configs/data.yaml` + `build_features.py`) y se propaga con `dvc repro`.
- El test set queda fijo durante toda la Fase 3; el propio held-out set solo debería regenerarse si cambia el dataset crudo o la lógica de features, nunca para "ayudar" a que un modelo dé mejor métrica.
