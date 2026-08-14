# 12. Reentrenamiento y promoción (Fase 6)

**Fecha:** 2026-08-14
**Estado:** Aceptada
**Fase:** 6 — Reentrenamiento y promoción

## Contexto

Fase 4 (ADR 0010) deja `run_training.py` registrando una versión nueva de
`churn-lightgbm` en el MLflow Model Registry cada vez que corre, pero
ninguna se promueve nunca — cita textual de ADR 0010: "todas las versiones
registradas son candidatas sin promover". Fase 5 (ADR 0011) deja
`run_monitoring.py` detectando y midiendo drift, pero explícitamente no
decide si ese nivel de drift amerita reentrenar — no-goal declarado a
propósito para no mezclar detección con la decisión de acción. Fase 6
necesita cerrar ese loop: decidir cuándo reentrenar, reentrenar reusando
hiperparámetros, y decidir si el resultado se promueve a producción.

Decisiones tomadas de antemano con el usuario, no relitigadas acá: el
trigger de reentreno es un script manual (no un orquestador automático
end-to-end, mismo patrón que Fase 3/5), y el reentreno reusa los últimos
hiperparámetros ganadores de Optuna en vez de re-correr la búsqueda
completa por defecto.

## Decisión

- **Aliases del MLflow Model Registry, no stages**: el proyecto resuelve
  `mlflow==3.15.1`; la API de stages (`transition_model_version_stage`)
  está deprecada desde MLflow 2.9, reemplazada por aliases
  (`set_registered_model_alias`/`get_model_version_by_alias`). Se usa el
  alias `champion` para marcar la versión de `churn-lightgbm` actualmente en
  producción. `README.md` describía el modelo viejo (Staging→Production) —
  corregido como parte de esta fase.
- **Los bytes del pipeline (champion y challenger) se leen de archivos
  locales versionados por DVC, no se descargan del Model Registry de
  MLflow.** Decisión revisada durante esta misma fase: la primera
  implementación reconstruía el `Pipeline` desde los artifacts del run de
  origen (`mlflow.sklearn.load_model`/`mlflow.lightgbm.load_model` vía
  `runs:/<run_id>/...`), pero el proxy de artifacts de MLflow en DagsHub
  resultó no ser confiable para descargas (ver "Consecuencias" — bug
  reproducido y documentado, no arreglable desde este repo). Como el
  `.pkl` del challenger ya queda en disco apenas termina
  `run_training.py`/`run_retrain.py`
  (`configs/training.yaml:model_output_path`), `run_promotion.py` lo lee
  directo de ahí. Para el champion — que es de un momento anterior — se
  mantiene una copia estable (`configs/registry.yaml:
  champion_pipeline_path`, `models/champion_pipeline.pkl`), actualizada
  por `run_promotion.py` cada vez que promueve, versionada por DVC vía
  `dvc add` (no vía stage, ver más abajo). El Model Registry + el alias
  `champion` siguen siendo la fuente de verdad de **metadata** (qué
  versión es cuál, historial, UI) — dejan de ser el mecanismo para traer
  los **bytes** del modelo. Es una extensión del mismo principio de ADR
  0010 (no acoplar el camino crítico a que el servidor de MLflow esté
  disponible), aplicado ahora también a la promoción, no solo al arranque
  de un futuro serving.
- **`models/champion_pipeline.pkl` se trackea con `dvc add`, no como
  `outs` de un stage de `dvc.yaml`** — es el primer archivo de este tipo
  en el proyecto (mismo mecanismo que `data/raw/telco.csv.dvc`, pero hasta
  ahora todo lo demás en `models/`/`data/processed/` se trackeaba vía
  pipeline). Tiene sentido porque no es la salida determinista de un
  stage — lo escribe una decisión de promoción, no una transformación de
  datos. `.gitignore` pasa de ignorar toda la carpeta `models/` a ignorar
  específicamente `*.pkl`/`*.json`, para que el futuro
  `champion_pipeline.pkl.dvc` sí quede trackeado por git.
- **`training/run_retrain.py` (nuevo) en vez de `dvc repro train_model`**
  para el camino default de reentreno-por-drift: `train_model` siempre
  corre Optuna completo (50 trials × 5 folds), lo cual contradice "no
  re-correr Optuna por defecto". El nuevo entrypoint reusa las funciones ya
  existentes de `run_training.py` (`refit_best_pipeline`,
  `save_pipeline_artifact`, `configure_mlflow`) pero se salta
  `run_optuna_search`, leyendo en cambio `models/best_params.json`
  (`fixed_params` + `study.best_params`), persistido aditivamente por
  `run_training.py::save_best_params` en cada corrida de tuning completo.
  `dvc repro train_model` queda intacto, reservado para forzar un re-tune
  completo a mano.
- **Trigger de reentreno vive en `monitoring/`**, no en `registry/` —
  CLAUDE.md ya asigna "Evidently AI drift detection + retraining trigger" a
  esa capa. `configs/retraining.yaml` es un config propio, deliberadamente
  separado de `configs/monitoring.yaml:drift_share` (ADR 0011 ya estableció
  que ese valor es sensibilidad de *detección*, no umbral de *acción*).
  `monitoring/retrain_trigger.py` (puro) decide sobre el resumen cross-batch
  de Fase 5 usando el **máximo** de `dataset_drift_share` en todo el
  resumen (no la última fila, porque el CSV es un barrido sintético
  completo por intensidad, no una serie temporal ordenada);
  `monitoring/run_retrain_check.py` es el orquestador/entrypoint, que solo
  informa — no reentrena. `retrain_threshold: 0.10` fue calibrado contra los
  resultados reales de Fase 5 (que nunca superaron 0.128 incluso a
  intensidad máxima) para que el trigger sea demostrable con los datos ya
  generados.
- **Comparación champion/challenger no solo por F1 agregado, también por
  segmento**: `registry/segment_evaluation.py` calcula
  `compute_classification_metrics` (reusada de `training/evaluation.py`)
  restringido a cada valor de `Contract`, `Payment Method` e
  `Internet Type` — las tres categóricas de baja cardinalidad que el EDA
  (`notebooks/01_eda.ipynb`, sección 6.1/8) identificó con el impacto
  diferencial más claro en churn. `registry/comparison.py::decide_promotion`
  promueve solo si el challenger gana el agregado por
  `promotion_margin=0.01` Y no retrocede en ningún segmento más de
  `segment_regression_tolerance=0.03` (más laxo que el margen agregado
  porque los segmentos tienen menos filas y más ruido de muestra). Ambos
  valores son propuestos, no derivados de un análisis de varianza real de
  F1 entre corridas de este proyecto — recalibrar si hace falta.
- **Bootstrap explícito**: la primera vez que corre `run_promotion.py` no
  existe ningún alias `champion` todavía. `get_champion_version` devuelve
  `None` explícitamente (no lanza una excepción genérica);
  `run_promotion.py` lo chequea con un `if` antes de construir cualquier
  `PromotionDecision` y promueve el challenger sin comparar — no hay contra
  qué comparar.
- **`registry/run_promotion.py` y `monitoring/run_retrain_check.py` no son
  stages de `dvc.yaml`** — mismo precedente que Fase 3/5 (ADR 0009/0011):
  son scripts de decisión corridos a mano, y nada todavía consume su output
  como dependencia de otro stage. `train_model` se modifica mínimamente: un
  output más (`models/best_params.json`).

## Alternativas consideradas

- **Agregar un flag `--skip-tuning` a `run_training.py`** en vez de un
  entrypoint nuevo — rechazada: cambiaría el contrato/comportamiento
  condicional de un entrypoint que `dvc.yaml` ya invoca directamente, más
  difícil de razonar que un script separado y explícito.
- **Trigger de reentreno en `registry/`** — rechazada: CLAUDE.md ya lo
  asigna a `monitoring/`, y el trigger no necesita ninguna dependencia de
  `registry/` (la cadena de capas `Training → Registry/Serving →
  Monitoring` sí permitiría que `monitoring/` importe de `registry/`, pero
  este script en particular no lo necesita).
- **API de stages (`transition_model_version_stage`)** — rechazada:
  deprecada en MLflow 3.x; aliases es la API recomendada.
- **Guardar el `Pipeline` completo en el Registry** (en vez de solo el
  `LGBMClassifier`) — fuera de alcance de Fase 6, cambiaría el contrato ya
  fijado en Fase 4 (ADR 0010); tampoco hubiera evitado el bug de descarga
  de DagsHub, que afecta a cualquier artifact, no solo al `LGBMClassifier`.
- **Downgrade del cliente `mlflow`** para evitar el bug de descarga de
  DagsHub — probado (`3.15.1` → `2.19.0`) y descartado: mismo error 500 en
  ambas versiones, así que no es un problema de compatibilidad de cliente.
- **Acceso directo al storage S3-compatible de DagsHub**, bypaseando el
  proxy de MLflow — no existe como opción documentada/soportada para
  artifacts de MLflow (sí para buckets DVC externos configurados a mano,
  que es un mecanismo distinto).
- **Esperar a que DagsHub resuelva el bug antes de promover algo** —
  rechazada: dejaría Fase 6 sin poder promover ningún modelo
  indefinidamente, sin fecha cierta de resolución, sobre infraestructura
  externa que ya demostró ser poco confiable en este punto específico.
- **Orquestador automático end-to-end** (detectar drift → reentrenar →
  promover en un solo comando) — rechazada por decisión previa del usuario:
  mantiene el patrón manual ya establecido en Fase 3/5, sin agregar el
  riesgo de reentrenar/promover sin supervisión en un proyecto de
  portafolio individual.

## Consecuencias

- **Historial del bug que motivó el rediseño**: la primera implementación
  de esta fase (reconstruir el pipeline vía descarga de artifacts de
  MLflow) quedó bloqueada al verificarla contra el servidor real de
  DagsHub — `load_pipeline_from_run` fallaba al descargar los artifacts
  (`preprocessor`/`model`) de cualquier run, tanto de runs de ese día como
  de un run de Fase 4 de varios días antes. Diagnóstico:
  `GET /api/2.0/mlflow/artifacts/list` en el servidor de DagsHub devolvía
  `200 OK` pero sin la clave `"files"` en el JSON (solo `root_uri`),
  aunque los archivos eran visibles en la UI cruda de MLflow; la descarga
  vía `/api/2.0/mlflow-artifacts/artifacts/<path>` devolvía `500` de forma
  consistente. Se descartaron como causa: cuota de storage, credenciales
  (token de DagsHub válido), e incompatibilidad de versión del cliente
  (mismo error con `mlflow==3.15.1` y `mlflow==2.19.0`) — ver alternativas
  consideradas. Reporte completo con evidencia técnica en
  `dagshub-artifact-bug-report.md` (fuera del repo, en las notas de
  aprendizaje del proyecto), pendiente de mandar a soporte de DagsHub. El
  rediseño a lectura local/DVC (ver "Decisión") deja Fase 6 funcional sin
  depender de que ese bug se resuelva — **verificado end-to-end contra el
  servidor real de DagsHub el mismo día**: bootstrap (versión 3 de
  `churn-lightgbm` promovida a `champion` sin champion previo) y comparación
  (challenger == champion, `delta agregado=0.0000`, correctamente no
  promovido) corrieron sin error, con `models/champion_pipeline.pkl`
  generado y `dvc add`+`dvc push` hechos después de correr esto.
- `registry/` tiene contenido real por primera vez.
- `models/` gana dos outputs nuevos: `best_params.json` (trackeado por DVC
  vía el stage `train_model`) y `champion_pipeline.pkl` (trackeado por DVC
  vía `dvc add`, actualizado por `run_promotion.py` en cada promoción — ver
  "Decisión"). Después de correr `run_promotion.py` con una promoción real,
  hace falta `dvc add models/champion_pipeline.pkl && dvc push` a mano para
  dejarlo versionado y disponible para otra máquina — mismo tipo de
  fricción que `dvc commit train_model` tras `run_retrain.py`.
- Correr `run_retrain.py` fuera de `dvc repro` sobrescribe
  `models/lightgbm_pipeline.pkl` por fuera del control de DVC — después
  hace falta `dvc commit train_model` para resincronizar `dvc.lock`, mismo
  tipo de fricción que ya existe en Fase 3/5 al correr scripts fuera de
  `dvc.yaml`.
- El experimento MLflow `phase6-promotion` queda como el registro histórico
  de cada decisión de promoción, incluyendo las métricas por segmento como
  artifacts JSON. En git solo se versiona el resumen
  (`reports/registry/promotion_results.csv` + `promotion_summary.md`),
  mismo criterio que `reports/model_selection_summary.md` de Fase 3 y
  `reports/monitoring/drift_monitoring_summary.md` de Fase 5.
- `README.md` actualizado: la sección del Registry ahora describe aliases
  en vez del modelo viejo de stages (Staging→Production).
