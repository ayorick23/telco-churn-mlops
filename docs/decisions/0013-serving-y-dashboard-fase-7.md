# 13. Serving y dashboard (Fase 7)

**Fecha:** 2026-08-16
**Estado:** Aceptada
**Fase:** 7 — Serving + dashboard

## Contexto

Fase 6 (ADR 0012) dejó el champion resuelto de una forma específica:
los bytes del `Pipeline` se leen de `models/champion_pipeline.pkl`, un
archivo local versionado por DVC — no se descargan del Model Registry de
MLflow (bug de descarga en el proxy de artifacts de DagsHub). El Registry +
alias `champion` siguen siendo la fuente de verdad de *metadata*. Fase 7
necesita exponer ese modelo vía una API REST (FastAPI) con explicabilidad
SHAP, y un dashboard (Streamlit) con predicciones, estado de producción,
drift e historial de reentrenamiento — sin tocar Docker/despliegue (Fase 8)
ni introducir orquestación automática (ya rechazada en ADR 0012).

Antes de implementar se resolvieron cuatro decisiones de diseño con el
usuario (AskUserQuestion) y una quinta técnica surgió durante la
implementación (bug de Streamlit magic commands).

## Decisión

- **El dashboard lee `reports/*.md`/`*.csv` directo** (ya versionados en git
  por Fases 3/5/6), en vez de agregar endpoints nuevos a la API solo para
  mostrarlos. `configs/dashboard.yaml:report_configs` apunta a los configs
  existentes (`model_selection.yaml`/`monitoring.yaml`/`registry.yaml`) para
  no duplicar `reports_dir` una tercera vez.
- **SHAP en un endpoint separado `POST /explain`**, no como flag de
  `/predict` — mantiene `/predict` rápido por default. `/predict` y
  `/explain` **nunca dependen de que MLflow esté arriba**: el `Pipeline` se
  carga una sola vez por proceso desde el archivo local
  (`serving/api/model_loader.py::get_pipeline`, reusando
  `registry/run_promotion.py::load_local_pipeline` en vez de duplicar el
  `joblib.load`). La versión del champion que devuelven es la resuelta al
  arrancar el proceso (lifespan de FastAPI) — si el Registry no está
  disponible en ese momento, siguen sirviendo con
  `champion_version="unknown"`. Solo `GET /model-info` pega contra el
  Registry en cada llamada, porque su propósito es mostrar el estado vivo.
- **El schema de input de la API (`CustomerFeatures`, Pydantic) espera los
  37 campos crudos del cliente** (mismos nombres que `data/schema.py`,
  incluyendo `City` aunque el champion actual la ignore vía
  `remainder='drop'` — mantiene el contrato honesto ante un futuro cambio de
  familia de modelo). `serving/api/features.py` reusa directo
  `features/build_features.py::add_engineered_features` para derivar
  `is_new_customer`/`num_extra_services` — cero lógica de feature
  engineering duplicada entre Fase 2 y Fase 7.
- **Docker se difiere a Fase 8**: `docker-compose.yml` queda vacío, mismo
  patrón que "sin CI/CD hasta Fase 8" confirmado tras Fase 2. La API y el
  dashboard corren con `uv run uvicorn ...`/`uv run streamlit run ...`.
- **El dashboard incluye botones de acción** en una sección "Operaciones /
  Simulación" separada de las vistas de solo-lectura: correr
  `monitoring.run_monitoring`, `monitoring.run_retrain_check`,
  `training.run_retrain` y `registry.run_promotion` vía `subprocess`,
  streameando stdout. Reentrenar/promover requieren un checkbox de
  confirmación explícito antes de habilitar el botón. Decisión de portafolio
  (recomendada y aceptada): un proyecto individual sin tráfico real que
  proteger se demuestra mejor con el loop completo disparable en vivo que
  con capturas estáticas; sigue siendo una acción manual por click humano,
  mismo nivel de supervisión que ADR 0012 exige — lo que ADR 0012 rechaza es
  un orquestador *automático* encadenado, no un botón. La UI deja explícito
  que estas acciones sobreescriben archivos locales que necesitan
  `dvc add`/`dvc push` a mano después para persistir.

## Alternativas consideradas

- **Endpoints nuevos en la API para servir reportes** — rechazada: los
  reportes ya son archivos versionados en git, agregar una capa HTTP encima
  solo para leerlos es trabajo sin beneficio mientras nada más los consuma.
- **SHAP incluido en `/predict` vía flag** — rechazada: acoplaría la
  latencia de SHAP (TreeExplainer) al endpoint de uso más frecuente.
- **Dashboard de solo lectura, sin botones de acción** — considerada y
  descartada a favor de la opción con botones tras discutirlo con el
  usuario: el valor de portafolio de un demo en vivo pesa más que el riesgo,
  inexistente acá (sin usuarios reales, sin tráfico de producción).
- **`CustomerFeatures(**values)` con un `dict[str, object]` dinámico** en el
  formulario del dashboard — rechazada tras fallar mypy (no puede validar
  `**dict` dinámico contra kwargs tipados); se usa
  `CustomerFeatures.model_validate(values)` en su lugar, semánticamente
  idéntico y mypy-limpio.

## Consecuencias

- `serving/api/` y `dashboard/` tienen contenido real por primera vez.
  Nuevos configs `configs/serving.yaml` y `configs/dashboard.yaml`, mismo
  patrón de un config por script ya establecido.
- **Bug de Streamlit "magic commands" encontrado y corregido durante la
  verificación manual en browser**: una expresión ternaria usada como
  statement (`st.markdown(x) if x else st.info(y)`) hace que Streamlit
  auto-renderice el `DeltaGenerator` devuelto por la rama ejecutada, además
  del contenido real — se ve como texto de "basura" (`DeltaGenerator(...)`)
  encima del reporte. Corregido reemplazando por `if`/`else` explícito en
  `render_production_tab`/`render_drift_tab` (`dashboard/app.py`). Lección
  para código Streamlit futuro en este proyecto: nunca usar una expresión
  ternaria de dos llamadas `st.*` como statement de nivel superior de una
  función — usar `if`/`else` siempre que ambas ramas tengan efecto de
  render.
- Verificado end-to-end en browser real (no solo tests): formulario de
  predicción de 37 campos generado dinámicamente desde `CustomerFeatures`
  (`dashboard/form_fields.py::build_field_specs`, introspección de Pydantic
  — evita declarar los campos dos veces), `/predict`+`/explain` contra el
  champion real (versión 3), gráfico de SHAP, tabs de Producción/Drift
  leyendo los reportes reales de Fases 3/5/6, y el botón de
  `run_retrain_check` ejecutando el script real y mostrando su stdout real
  (`dataset_drift_share=0.1282 vs. umbral=0.1` → "Retrain recomendado: SI").
- 26 tests nuevos (`tests/unit/serving/api/`, `tests/unit/dashboard/`),
  mockeando `get_pipeline`/`get_champion_metadata`/`explain_prediction` en
  los tests de la API (nunca cargan el `.pkl` real ni pegan contra MLflow) y
  cubriendo solo las funciones puras del dashboard (`form_fields.py`,
  `reports.py`, `api_client.py`), no la UI de Streamlit en sí.
- Nuevas dependencias directas: `requests` (dashboard → API) y `httpx`
  (requerido por `TestClient` de FastAPI/Starlette) — ambas ya eran
  transitivas, mismo patrón que `pyarrow`/`joblib`/`python-dotenv` en fases
  previas.

## Actualización — ronda de UX del dashboard (2026-08-16, misma fase)

Tras la verificación manual inicial, el usuario pidió una ronda de mejoras
visuales sobre el dashboard ya funcional. Decisiones tomadas (todas
verificadas en browser real, no solo con tests):

- **Navegación por sidebar** vía `st.navigation`/`st.Page` (reemplaza
  `st.tabs`) + espacio para logo vía `st.logo` (opcional — si
  `configs/dashboard.yaml:logo_path` no existe en disco, el dashboard sigue
  funcionando sin logo, no rompe).
- **`City` pasa de texto libre a un select acotado a las ciudades reales**
  de `data/processed/train.parquet` (`dashboard/data_sources.py`) — pedido
  explícito del usuario: un valor inventado rompería la relación con el
  resto de las features aunque el champion actual la ignore vía
  `remainder='drop'` (ADR 0009); una futura familia (CatBoost) sí podría
  usarla nativa.
- **Formulario de predicción sin `st.form`**: dentro de un form los widgets
  no reaccionan entre sí hasta el submit, y acá hace falta reactividad en
  vivo (elegir "Internet Service: No" bloquea de inmediato los 8 campos que
  dependen de tener internet). `dashboard/prediction_form.py` concentra el
  motor de estado — puro, sin importar `streamlit`, testeable sin runtime:
  `apply_dependency_rules`/`locked_fields` (reglas: sin Internet Service
  bloquea sus 8 addons; sin Dependents → 0; sin Phone Service → Multiple
  Lines=No; sin Referred a Friend → 0 referidos; Under 30 siempre derivado
  de Age, nunca editable directo), `random_value` (rangos reales de
  `train.parquet`, no inventados) y `to_payload` (traduce el sentinel de UI
  `NONE_SENTINEL` a `None` real recién al armar el request).
- **Errores de validación inline por campo**, no solo un mensaje general al
  final: se parsean los `errors()` de la `ValidationError` de Pydantic y se
  muestra un `st.caption` en rojo debajo del widget específico. Como los
  campos `Literal` ya son `selectbox` (no aceptan valores inválidos por
  diseño) y las reglas de dependencia bloquean la mayoría de combinaciones
  imposibles, lo que queda por señalar son sobre todo rangos numéricos.
- **3 cards de resultado** (`st.metric` en `st.columns(3)`: probabilidad,
  predicción, champion version) en vez de metric+metric+caption.
- **Nombres de feature del gráfico SHAP humanizados**
  (`dashboard/shap_labels.py::humanize_shap_feature_name`): los nombres que
  se veían (`categorical__Contract_Month-to-Month`) no son de FastAPI — son
  los que genera `ColumnTransformer.get_feature_names_out()` internamente;
  se tradujeron a `"Contract: Month-to-Month"` parseando contra la lista de
  columnas conocidas (no contra valores hardcodeados, para no asumir un
  placeholder de nulos específico).
- **Gráfico SHAP con color divergente** (rojo=aumenta riesgo de churn,
  azul=lo reduce) vía Altair (`st.altair_chart`), siguiendo la skill de
  dataviz del proyecto: SHAP es exactamente el caso de uso de un par
  divergente (magnitud con signo alrededor de un cero real), no una paleta
  categórica. Colores tomados del par divergente validado de la skill
  (`blue #2a78d6`/`red #e34948` en claro, `#3987e5`/`#e66767` en oscuro),
  seleccionados vía `st.get_option("theme.base")`.
- **Producción y Drift: solo `st.dataframe`, nunca markdown+dataframe de la
  misma tabla** — los `.md` de Fase 3/5/6 traen la tabla comparativa
  embebida como texto; se dejó de renderizarla ahí (`_extract_narrative`
  corta el texto justo antes de la primera línea que empieza con `"|"`) y
  se muestra siempre como `st.dataframe` desde el `.csv` correspondiente.
  Drift no tenía narrativa (`drift_monitoring_summary.md` es solo
  título+tabla), así que ese `.md` se dejó de leer directamente.
- **Iconos en los botones de Operaciones** (`st.button(icon=...)`) +
  `st.status(...)` en vez de `st.spinner`+`st.code` suelto (da una caja
  colapsable "Corriendo → ✅ Listo" con el output adentro) + `st.toast()` al
  terminar cada acción.
- **Formulario agrupado en `st.expander` por categoría** (Datos personales /
  Cuenta y contrato / Servicios / Cargos —
  `prediction_form.py:FIELD_GROUPS`) en vez de una grilla plana de 37
  campos — sugerencia propia aceptada por el usuario junto con el resto del
  paquete de UX.
- Nueva dependencia directa: `altair` (ya transitiva vía `streamlit`, ahora
  importada directo para el gráfico de SHAP — mismo patrón que
  `requests`/`httpx`).

## Actualización — pisos mínimos numéricos (2026-08-16, misma fase)

El usuario detectó que "Referred a Friend: Yes" con "Number of Referrals: 0"
(y el mismo problema con Dependents) era una combinación aceptada por el
formulario pero inexistente en el dataset real. Verificado contra
`data/processed/train.parquet` antes de corregir (no se asumió el alcance):
`Referred a Friend`/`Dependents` en "Yes" nunca tienen su contador en 0
(mínimo real 1); `Monthly Charge`/`Total Charges`/`Total Revenue` nunca son
0 para ningún cliente (piso fijo, incondicional); `Avg`/`Total Long Distance
Charges` son 0 solo cuando `Phone Service=No` (si no, mínimo ~1);
`Avg Monthly GB Download` es 0 solo cuando `Internet Service=No`. En
cambio, `Total Refunds`/`Total Extra Data Charges` son 0 en ~90-92% de las
filas reales — se dejaron sin piso, a propósito, para no sobrecorregir.

`prediction_form.py` gana `CONDITIONAL_MINIMUMS`/`UNCONDITIONAL_MINIMUMS` +
`numeric_min_value()`. `apply_dependency_rules()` sube el valor al piso
correspondiente si quedó en 0 pero la condición que lo habilita está
activa (mismo mecanismo que ya fuerza a 0 en la dirección contraria);
`app.py` pasa ese piso como `min_value` a cada `st.number_input`, así que
la UI no permite bajar de ahí ni con el stepper ni escribiendo a mano —
no es solo una corrección post-hoc, es un límite duro en el widget.
