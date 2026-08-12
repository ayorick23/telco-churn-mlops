# 10. Pipeline de entrenamiento definitivo (Fase 4): tuning, stage de DVC y Registry

**Fecha:** 2026-08-12
**Estado:** Aceptada
**Fase:** 4 — Training pipeline definitivo

## Contexto

Fase 3 (ADR 0009) comparó 4 familias con hiperparámetros por defecto y determinó
que LightGBM gana por F1 de la clase positiva (0.9290 sobre `test.parquet`). Fase 4
deja de comparar familias y construye el pipeline de producción para la única
familia ganadora: tuning con Optuna (ya anotado como pendiente en `pyproject.toml`),
integración a `dvc.yaml`, y qué tan formalmente se registra el modelo resultante.
Había que decidir: estrategia de validación para el tuning, cómo separar el config
de esta fase del de Fase 3, qué produce el nuevo stage de DVC como output, y si el
modelo se registra en el MLflow Model Registry ya en esta fase.

## Decisión

- **Validación de Optuna**: Stratified K-Fold (5 folds) sobre `train.parquet`,
  optimizando el promedio de F1 de la clase positiva entre folds. `test.parquet`
  permanece completamente fuera del proceso de tuning — solo se usa una vez, al
  final, para evaluar el pipeline con los mejores hiperparámetros encontrados. Se
  prefiere sobre un holdout fijo adicional dentro de train porque da una estimación
  más estable con ~5.6k filas y evita gastar aún más filas de train en un segundo
  split.
- **Config**: el `configs/training.yaml` de Fase 3 se renombra a
  `configs/model_selection.yaml` (sigue siendo el config de
  `run_model_selection.py`, que sigue corriéndose a mano, fuera de `dvc.yaml`). Se
  crea un `configs/training.yaml` nuevo, scopeado a Fase 4, con el espacio de
  búsqueda de Optuna para LightGBM, número de trials, folds de CV, y el nombre del
  modelo registrado. Sigue el mismo patrón que `data.yaml`/`features.yaml`: un
  archivo de config por stage de `dvc.yaml`.
- **Nuevo stage `train_model` en `dvc.yaml`**: depende de `data/processed/train.parquet`,
  `test.parquet` y el módulo de entrenamiento final; como output serializa el
  `Pipeline` fiteado (preprocesador + `LGBMClassifier` con los mejores
  hiperparámetros) a `models/lightgbm_pipeline.pkl`, trackeado por DVC. Además de
  ese artefacto local, el run se loguea a MLflow igual que en Fase 3 (params,
  métricas, modelo vía `mlflow.lightgbm`). Se guardan ambos porque cumplen roles
  distintos: el `.pkl` versionado por DVC da reproducibilidad offline atada al
  commit exacto de datos+código (`dvc repro` sin depender de que el server de
  MLflow esté arriba); el run de MLflow es el registro histórico de experimentos y
  lo que alimenta al Model Registry.
- **Se registra en el MLflow Model Registry ya en Fase 4**, con
  `registered_model_name="churn-lightgbm"` al loguear el modelo del run final. Esto
  deja una versión nueva en el Registry cada vez que corre el stage. La lógica de
  comparación champion/challenger y promoción entre versiones (`registry/`) se
  implementa recién en Fase 6 — Fase 4 solo registra, no promueve ni compara contra
  producción.

## Alternativas consideradas

- **Holdout fijo adicional para Optuna** en vez de K-Fold — rechazada: con el
  dataset de este tamaño, un solo holdout dentro de train tiene más varianza de
  trial a trial y además reduce aún más las filas disponibles para fitear en cada
  trial.
- **Mantener un solo `training.yaml`** agregando una sección para Fase 4 — rechazada
  por la misma razón que llevó a separar `data.yaml`/`features.yaml`: mezclar el
  config de un script exploratorio (Fase 3, corrido a mano) con el de un stage de
  producción de `dvc.yaml` (Fase 4) hace más frágil el `params` tracking de DVC, que
  lee el archivo completo por stage.
- **Solo loguear a MLflow, sin artefacto local en `models/`** — rechazada: acoplaría
  `dvc repro` a que el servidor de MLflow de DagsHub esté disponible para poder
  reproducir el pipeline completo localmente; el `.pkl` versionado por DVC es
  necesario además para Fase 7 (serving), que debe poder cargar el modelo sin
  depender de un registry remoto en el camino crítico de arranque.
- **Posponer el registro en el Model Registry a Fase 6** — rechazada: registrar
  desde ya es de bajo costo (un parámetro más en `log_model`) y deja versiones
  reales en el Registry para que Fase 6 pueda implementar y probar la lógica de
  promoción contra datos existentes, en vez de tener que generarlos recién ahí.

## Consecuencias

- `run_model_selection.py` (Fase 3) queda intacto salvo por el nombre del config
  que lee (`model_selection.yaml`); no se toca su lógica.
- Se agrega `models/` como carpeta trackeada por DVC (nuevo `.gitignore`/`.dvc`
  pointer), análoga a `data/processed/`.
- El módulo de tuning (Optuna + K-Fold) es reutilizable en Fase 6 si el reentreno
  automático necesita re-tunear en vez de reusar los últimos hiperparámetros
  ganadores.
- Cada corrida de `dvc repro train_model` crea una versión nueva en
  `churn-lightgbm` en el Registry, incluso si el resultado es peor que la versión
  anterior — Fase 6 es la que decide cuál queda como `Production`/`Champion`; hasta
  entonces, todas las versiones registradas son candidatas sin promover.
