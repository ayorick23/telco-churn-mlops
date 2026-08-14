# 11. Simulación de drift y monitoreo (Fase 5)

**Fecha:** 2026-08-13
**Estado:** Aceptada
**Fase:** 5 — Simulación de drift + monitoreo

## Contexto

Fase 4 (ADR 0010) dejó un pipeline de entrenamiento reproducible para LightGBM,
registrado en el MLflow Model Registry. Fase 5 necesita demostrar detección de
drift, pero no existe una segunda captura real de datos del telco — el dataset de
Kaggle es una foto estática, ya particionada en `train.parquet`/`test.parquet`
desde Fase 2 (ADR 0008). Había que decidir: cómo simular "datos futuros" sin un
segundo dataset real, dónde vive el código de inyección de drift, si la
generación de batches y la detección se integran a `dvc.yaml`, dónde queda el
resultado de la detección, y hasta dónde llega el alcance de esta fase (Fase 6 es
"Reentrenamiento y promoción" — el disparo de reentrenamiento no es de acá).

## Decisión

- **Simulación por perturbación de `train.parquet`**: se muestrean filas de
  `train.parquet` (con reemplazo) y se les aplica una perturbación configurable
  a un subconjunto de columnas, con una intensidad creciente entre 0.0 (batch de
  control, idéntico en distribución a la referencia) y 1.0 (perturbación
  completa). La historia simulada es "suba de precios + shift hacia contratos
  mensuales": `Monthly Charge` y `Tenure in Months` reciben un shift numérico de
  media; `Contract` y `Payment Method` se re-samplean hacia una distribución de
  categorías objetivo. Nunca se perturba `Churn Label` (el target) — reforzado en
  código, no solo por omisión, para no reintroducir leakage (ADR 0006).
- **`data/drift_simulation.py`**: la inyección de drift vive en la capa `data/`
  (no en `monitoring/`), consistente con la responsabilidad que ya le asigna el
  diagrama de arquitectura de `CLAUDE.md`. Es un módulo puro y determinista
  (`random_state` fijo) — sin MLflow, sin Evidently, sin I/O — para que la
  matemática de la perturbación sea testeable de forma aislada.
- **`monitoring/drift_detection.py` + `monitoring/run_monitoring.py`**: la
  detección usa Evidently AI (`Report` + `DataDriftPreset`, API de la versión
  0.7.x realmente instalada — ver "Consecuencias") comparando cada batch
  sintético contra `train.parquet` como referencia. Por cada batch se loguea un
  run de MLflow separado (params: nombre/intensidad del batch, umbral de
  `drift_share`; métricas: share de columnas con drift y conteo; artifacts:
  reporte HTML + JSON completo de Evidently). Un run final `"summary"` loguea una
  tabla cross-batch (CSV + Markdown) en `reports/monitoring/`, mismo patrón que
  el run `"summary"` de `run_model_selection.py` en Fase 3.
- **Ni la generación de batches ni la detección son stages de `dvc.yaml`**: se
  corren a mano (`uv run python -m churn_mlops.monitoring.run_monitoring`), mismo
  precedente que `run_model_selection.py` en Fase 3 (ADR 0009). Los batches son
  sintéticos y deterministas — no hace falta el tracking de staleness por hash de
  DVC para saber cuándo regenerarlos — y nada consume el resultado del monitoreo
  como dependencia de otro stage todavía.
- **Batches en memoria, no persistidos a disco**: se generan dentro de
  `run_monitoring.py`, se usan para la detección, y se descartan. No se agrega
  nada nuevo en `data/batches/` en esta fase, aunque el `.gitignore` ya reserva
  esa carpeta.
- **No-goal explícito**: Fase 5 no decide si un nivel de drift amerita
  reentrenar. `drift_share` en `configs/monitoring.yaml` es un parámetro de
  sensibilidad de *detección* de `DataDriftPreset` (a partir de qué proporción de
  columnas individualmente "drifted" Evidently marca el dataset completo como
  drifted) — no es un umbral de reentrenamiento. Esa lógica de trigger y
  promoción es de Fase 6 (`registry/`).

## Alternativas consideradas

- **Usar un segundo dataset real/externo** como proxy de "datos nuevos" —
  rechazada: no existe una segunda captura real de este telco, y un dataset
  externo no relacionado introduciría drift no interpretable, sin la historia de
  negocio explicable que sí da perturbar el propio dataset a propósito.
- **Persistir los batches sintéticos en `data/batches/`** — rechazada por ahora:
  son deterministas (mismo `random_state` + misma intensidad reproducen el mismo
  batch), así que no aportan nada nuevo guardados a disco en esta fase; agregar
  esa carpeta es trabajo extra sin un consumidor todavía. Si Fase 6 o una demo de
  Fase 7 necesita batches materializados para inspección o reuso, se reconsidera
  ahí.
- **Integrar generación de batches y/o detección a `dvc.yaml`** — rechazada: DVC
  versiona pipelines deterministas de producción con dependencias reales entre
  stages; acá no hay otro stage que consuma el output todavía, y el "cuándo
  regenerar" ya lo resuelve el `random_state` fijo, no el hash de DVC.
- **Implementar ya un trigger de reentrenamiento** (aunque el reentrenamiento en
  sí se ejecute en Fase 6) — rechazada explícitamente: mezclar detección con
  decisión de acción en la misma fase difumina el límite Fase 5/Fase 6 que el
  roadmap del proyecto ya traza.

## Consecuencias

- `src/churn_mlops/monitoring/` tiene contenido real por primera vez —
  `__init__.py` estaba vacío desde el scaffold inicial del proyecto.
- `configs/monitoring.yaml` es el quinto config por-stage/script del proyecto,
  con un espacio de perturbación declarado por `type` (`numeric_shift` /
  `categorical_reweight`), mismo patrón de despacho que `optuna.search_space` en
  `configs/training.yaml` (`tuning.py::build_search_space`, Fase 4).
- `pyproject.toml` tenía `evidently>=0.4`, pero la versión efectivamente resuelta
  por `uv` es `0.7.21` — la API cambió por completo entre esas versiones
  (`Report`/`Dataset`/`Preset` reemplazó a `Dashboard`/`model_profile` de la
  v0.4). Se corrige el pin a `evidently>=0.7` para que quede reflejado en el
  código qué versión se usó realmente; sin este ajuste, una máquina nueva podría
  resolver una versión incompatible con este módulo.
- El módulo de simulación de drift (`data/drift_simulation.py`) es genérico
  sobre cualquier `DataFrame` de referencia — si Fase 6 necesita generar batches
  de prueba para validar la lógica de promoción champion/challenger, puede
  reusarlo sin cambios.
- El experimento MLflow `phase5-monitoring` queda como el registro histórico de
  cada corrida de detección, incluyendo el HTML/JSON completo de Evidently por
  batch como artifacts. En git solo se versiona el resumen cross-batch
  (`reports/monitoring/drift_monitoring_results.csv` +
  `drift_monitoring_summary.md`), igual criterio que
  `reports/model_selection_summary.md` de Fase 3 — el HTML/JSON por batch pesa
  varios MB cada uno y ya vive en MLflow, así que queda gitignored
  (`reports/monitoring/*.html`, `*.json`).
