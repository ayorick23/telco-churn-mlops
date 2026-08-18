# 14. Testing, CI/CD y despliegue (Fase 8)

**Fecha:** 2026-08-17
**Estado:** Aceptada
**Fase:** 8 — Testing, CI/CD, despliegue

## Contexto

La tabla de infraestructura de CLAUDE.md ("FastAPI en Render/Railway,
Streamlit en Hugging Face Spaces, GitHub Actions para CI+CD") viene del
commit de inicialización del proyecto (`ab16a7e`), escrita antes de que
existiera código — a diferencia del resto de las fases, nunca se evaluó
con el mismo criterio (ADR + decisión explícita con el usuario). Antes de
implementar, se discutieron tres preguntas de fondo:

1. **¿Separar API (Render) y dashboard (Hugging Face Spaces) en dos
   plataformas distintas es una buena práctica, o solo una elección de
   portafolio?** Es lo segundo, y vale la pena decirlo explícito: una
   empresa real pondría ambos servicios en la misma nube/VPC (red
   privada, un solo pipeline de despliegue, observabilidad unificada).
   El split se mantiene aquí por costo cero y visibilidad de portafolio
   (Hugging Face Spaces indexa en el ecosistema de la comunidad ML), no
   porque sea el patrón productivo — decisión consciente, no ingenua.
2. **¿Qué rol cumple Docker?** Empaquetado reproducible, no orquestación.
   Este proyecto ya tiene un precedente de dependencias nativas
   sensibles a versión (`llvmlite`/`numba` vía `shap`, ver sección "Known
   dependency quirk" de CLAUDE.md) — un Dockerfile fija exactamente el
   entorno una sola vez y esa misma imagen corre local, en CI, y en
   producción.
3. **¿Alcanza con lint+test en CI para "demostrar CI/CD", o hace falta
   más?** Alcanza si además hay un gate real (branch protection exigiendo
   el check) y un paso de CD explícito gateado en que CI haya pasado — no
   alcanza si el "CD" es solo el auto-deploy silencioso nativo de la
   plataforma, porque eso no está condicionado a que nada esté roto.

Se detectó además, verificando el repo antes de implementar, deuda
acumulada por no haber tenido CI desde el principio: `ruff check .` daba
23 errores (imports desordenados, en tests de casi todas las fases,
autofixable) y `mypy src/` daba 6 errores idénticos
(`sys.stdout.reconfigure` sin narrow de tipo, en los 6 scripts
`run_*.py` de Fases 3-6 que manejan el problema de encoding de la
consola de Windows). Se corrigieron como parte de levantar el gate, no
se dejaron como excepción — la deuda es evidencia concreta de que
integrar CI recién en la última fase tiene costo real: sin un gate desde
el día 1, esos errores se acumularon en vez de corregirse PR por PR.

## Decisión

- **Se mantiene el split de hosting** (API en Render, dashboard en
  Hugging Face Spaces) — decisión explícita del usuario tras entender el
  trade-off del punto 1 de arriba.
- **Solo la imagen de la API necesita el modelo.** El dashboard
  (`dashboard/api_client.py`) es un cliente HTTP puro de la API vía
  `CHURN_API_BASE_URL` (env var ya soportada desde Fase 7) — nunca carga
  el `.pkl` localmente. Su imagen (`docker/dashboard.Dockerfile`) no
  necesita credenciales de DagsHub en ningún momento.
- **La imagen de la API se construye una sola vez, en GitHub Actions, y
  se publica en GHCR** (`ghcr.io/ayorick23/telco-churn-mlops-api`,
  autenticado con el `GITHUB_TOKEN` que Actions ya provee — sin secret
  adicional). El job `deploy` corre `dvc pull
  models/champion_pipeline.pkl.dvc` (credenciales de DagsHub reusadas de
  `MLFLOW_TRACKING_USERNAME`/`PASSWORD`, mismo patrón que
  `.env.example`) antes de `docker build`, así `docker/api.Dockerfile`
  simplemente hace `COPY models/champion_pipeline.pkl` — evita pasarle
  credenciales de build a Render, que lo soporta peor que GitHub
  Actions.
- **Render despliega la imagen ya construida de GHCR** (servicio tipo
  "Existing Image", no build-from-Dockerfile) — el job `deploy` dispara
  el redeploy vía el Deploy Hook URL de Render (`RENDER_DEPLOY_HOOK_URL`,
  un POST sin body).
- **Hugging Face Spaces construye la imagen del dashboard él mismo**, vía
  un Space tipo **Docker** (reusa `docker/dashboard.Dockerfile` directo,
  no el SDK nativo de Streamlit — ese SDK espera `app.py` en la raíz del
  repo y `requirements.txt`, incompatible con el src-layout + uv de este
  proyecto). El job `deploy` empuja el repo al remote git del Space con
  `HF_TOKEN`.
- **CI y CD viven en un solo workflow**
  (`.github/workflows/ci-cd.yml`), job `ci` (siempre, push y PR) y job
  `deploy` (`needs: ci`, solo en push directo a `main` — nunca en PRs,
  incluso de forks, por diseño de GitHub Actions con `pull_request`).
  Un solo archivo es proporcional a la escala de un proyecto individual.
- **`ruff format` excluye `notebooks/`** (`[tool.ruff.format] exclude`,
  `pyproject.toml`) — mismo criterio que las excepciones de lint ya
  existentes para notebooks (`E402`/`F401`): son exploración de la Fase
  1, no producción, y reformatear celdas ya revisadas no aporta nada al
  gate de CI.
- **El redeploy de la API tras una promoción NO es automático** —
  consistente con que reentrenar/promover ya son manuales (ADR 0012) y
  con que los botones de "Operaciones" del dashboard (ADR 0013) ya
  avisan que hace falta `dvc add`/`dvc push` después de promover. El
  flujo completo pasa a ser: promover → `dvc push` → commit del `.dvc`
  pointer actualizado → push a `main` → el job `deploy` reconstruye y
  redespliega la API con el nuevo champion.
- **Sin servicio `mlflow` en `docker-compose.yml`** — el MLflow real ya
  vive en DagsHub; un servidor local duplicaría infraestructura sin caso
  de uso claro en este proyecto.

## Alternativas consideradas

- **Consolidar API+dashboard en una sola plataforma** (ambos como
  servicios Docker en Render, por ejemplo) — más representativo de cómo
  se estructura en una empresa (un solo proveedor, un solo pipeline),
  pero rechazada por decisión explícita del usuario: mantener el valor
  de portafolio/visibilidad de Hugging Face Spaces para el dashboard.
- **Que Render construya la imagen desde el Dockerfile del repo**
  (build-from-source nativo de la plataforma) — rechazada porque
  requeriría darle a Render credenciales de DagsHub como build secret,
  un mecanismo que soporta peor que GitHub Actions (que ya gestiona
  secrets nativamente); más simple publicar una imagen ya construida y
  que Render solo la despliegue.
- **`dvc pull` del modelo en el arranque del contenedor** (en vez de
  hornearlo en la imagen en build time) — rechazada: reintroduce una
  dependencia de red externa en el hot path de arranque, justo lo que
  ADR 0012/0013 evitan a propósito para el servido de predicciones
  (`/predict`/`/explain` nunca dependen de que un servicio externo esté
  arriba). Hornear el modelo en la imagen mantiene esa garantía también
  en producción — el costo es que promover un nuevo champion requiere un
  redeploy explícito, que ya es el comportamiento esperado (ver más
  arriba).
- **Silenciar los 23 errores de lint / 6 de mypy como excepciones
  conocidas** en vez de corregirlos — rechazada: son autofix/mecánicos,
  de bajo riesgo, y dejar el primer commit de esta fase con excepciones
  documentadas hubiera perpetuado exactamente el problema que motivó
  escribir este ADR.

## Consecuencias

- `docker/api.Dockerfile` y `docker/dashboard.Dockerfile` tienen
  contenido real por primera vez (antes solo `docker/.gitkeep`), igual
  `docker-compose.yml` y `.github/workflows/ci-cd.yml` (antes
  `.github/workflows/.gitkeep`).
- Nuevos secrets a cargar a mano en GitHub (Settings → Secrets and
  variables → Actions), no gestionables por este ADR ni por código:
  `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
  `MLFLOW_TRACKING_PASSWORD`, `RENDER_DEPLOY_HOOK_URL`, `HF_TOKEN`,
  `HF_SPACE_REMOTE`.
- Branch protection sobre `main` (exigir el status check `ci`) se
  configura a mano en GitHub — no es un artefacto versionable en el
  repo.
- El job `deploy` fallará hasta que el Web Service de Render y el Space
  de Hugging Face existan y los secrets estén cargados — se documenta
  como paso manual pendiente, no como un bug de la implementación.
- `tests/integration/` sigue sin tests reales (solo `__init__.py`) — no
  se agregó cobertura de integración en esta fase, fuera de alcance.

## Actualización — consolidación en Hugging Face Spaces (2026-08-18)

Al ejecutar el rollout real (los servicios de Render/HF nunca se habían
levantado, solo el código estaba listo), Render cambió su política de free
tier y ahora exige verificación con tarjeta de crédito antes de crear un
Web Service — no cobra, pero bloquea el flujo sin tarjeta. Decisión
explícita del usuario: en vez de agregar una tarjeta, **se abandona Render
y se consolidan API + dashboard como dos Spaces Docker independientes en
Hugging Face Spaces**, revirtiendo el punto 1 de la sección "Decisión"
original.

- **Se mantiene la razón de fondo de por qué la API nunca construye su
  propia imagen en la plataforma de hosting** (evitar darle credenciales
  de DagsHub a un build remoto) — eso no cambió, solo el destino final de
  la imagen ya construida. La imagen de la API se sigue publicando una
  sola vez en GHCR desde GitHub Actions (sin cambios en ese paso).
- **El Space de la API es un Dockerfile mínimo de una sola línea**
  (`FROM ghcr.io/ayorick23/telco-churn-mlops-api:latest`) — verificado que
  el paquete de GHCR es público (`ghcr.io/ayorick23/telco-churn-mlops-api`),
  así que el Space lo puede pullear sin secret ni login adicional.
- **Hueco descubierto al implementar esto**: el plan original de esta ADR
  decía que HF Spaces "reusa `docker/dashboard.Dockerfile` directo", pero
  el SDK Docker de HF Spaces exige un archivo llamado exactamente
  `Dockerfile` en la **raíz** del repo del Space — no soporta apuntar a
  una ruta custom vía metadata de `README.md`. Nunca se detectó porque el
  Space nunca se había creado. Se corrige generando, dentro del propio
  job `deploy` (no commiteado a `main`), un árbol mínimo por Space:
  - Space de la API: solo `Dockerfile` (la línea de arriba) + `README.md`
    con el front-matter YAML que Spaces requiere (`sdk: docker`,
    `app_port: 8000`).
  - Space del dashboard: `pyproject.toml`, `uv.lock`, `src/`, `configs/`,
    `reports/` (el contexto real que ya usaba
    `docker/dashboard.Dockerfile`) más una copia de ese Dockerfile como
    `Dockerfile` en la raíz y un `README.md` con `app_port: 8501`. No se
    empuja el monorepo completo (notebooks, docs, tests, data) — sin
    motivo para que viaje a un Space cuyo único trabajo es construir esa
    imagen.
  - `docker/api.Dockerfile` y `docker/dashboard.Dockerfile` no cambian —
    siguen siendo la fuente real para `docker-compose.yml` y el build de
    validación del job `ci`; el job `deploy` solo los reutiliza (copia o
    referencia) al armar el árbol de cada Space.
- **Secrets**: `RENDER_DEPLOY_HOOK_URL` se da de baja (ya no se usa).
  `HF_SPACE_REMOTE` se reemplaza por dos secrets separados,
  `HF_API_SPACE` (`ayorick23/telco-churn-mlops-api`) y
  `HF_DASHBOARD_SPACE` (`ayorick23/telco-churn-mlops-dashboard`). `HF_TOKEN`
  se reusa para ambos pushes.
- **Riesgo real que se acepta**: el free tier de HF Spaces duerme los
  Spaces inactivos. Antes solo el dashboard tenía ese riesgo (la API
  vivía en Render); ahora ambos pueden estar dormidos a la vez, así que
  un dashboard recién despertado puede pegarle a una API que también está
  arrancando — la primera predicción después de inactividad puede tardar
  o dar timeout. No se mitiga en esta fase (agregar un healthcheck/retry
  en el dashboard queda como mejora futura, no bloqueante para un
  proyecto de portafolio).

## Actualización — tests de integración (2026-08-18)

Se cierra el hueco documentado arriba en "Consecuencias" (`tests/integration/`
sin tests reales). Motivador directo: los dos bugs reales encontrados al
validar el rollout desde la PC de casa (commits `c006a49` y `a63c4bd`, ver la
actualización anterior) son exactamente la clase de bug que ni los tests
unitarios (todos mockean `get_pipeline`/`explain_prediction`, o fabrican el
archivo de ciudades con `tmp_path`) ni el paso "build sin push" del job `ci`
(solo construye, nunca corre el contenedor) podían atrapar.

Tres archivos nuevos en `tests/integration/`, cada uno cerrando un hueco
específico (detalle del razonamiento en el docstring de cada archivo):

- **`test_prediction_pipeline.py`**: el Pipeline champion real (sin mocks),
  vía `TestClient` — `/predict` y `/explain` contra el `.pkl` de verdad, no
  contra el `_FakePipeline` que usa `tests/unit/serving/api/test_main.py`.
- **`test_dashboard_data_sources.py`**: `load_known_cities()` contra el
  `reports/known_cities.json` **real** commiteado, no un archivo fabricado
  por el test (el unit test sí lo fabrica con `tmp_path`) — este test
  hubiera atrapado el bug de `a63c4bd` directo, sin necesitar Docker.
- **`test_docker_stack.py`**: `docker compose up --build` real de los dos
  servicios + smoke tests contra los contenedores vivos (`/health`,
  `/predict`, reachability del dashboard). Hubiera atrapado el bug de
  `libgomp1` (`c006a49`) porque, a diferencia del paso "build sin push" que
  existía antes, este sí **corre** el contenedor, no solo lo construye.

Decisiones de diseño:

- Los tres usan `skipif` en vez de fallar duro cuando falta un prerequisito
  (el `.pkl` local, o el daemon de Docker corriendo) — así `uv run pytest`
  sigue siendo seguro en cualquier máquina de desarrollo, sin bloquear a
  alguien que no corrió `dvc pull` o no tiene Docker Desktop abierto.
- El job `ci` se reordenó: `dvc pull` y `docker/setup-buildx-action` ahora
  corren **antes** de `uv run pytest` (antes iban después), porque los
  tests de integración necesitan el `.pkl` y Docker disponibles en ese
  punto. Los dos steps "build de la imagen... (sin push)" que existían se
  eliminaron — quedaron redundantes: el `docker compose up --build` de
  `test_docker_stack.py` ya construye ambas imágenes, y además las corre —
  estrictamente más cobertura que solo construirlas.
- `test_docker_stack.py` escribe un `.env` mínimo si no existe (necesario
  porque `docker-compose.yml` declara `env_file: .env` para el servicio
  `api`, y un checkout limpio de CI no lo tiene — está en `.gitignore`) y lo
  borra al terminar; si ya existe (dev local con credenciales reales) lo
  deja intacto. Seguro de escribir solo por la comprobación de "no existía
  antes" — nunca pisa un `.env` real de un desarrollador.

Consecuencias:

- El job `ci` ahora tarda más (construye ambas imágenes Docker completas
  dentro de `docker compose up --build`, en vez de solo dos builds
  livianos) — aceptado: es el costo directo de que CI valide algo que antes
  solo se verificaba a mano.
- `test_docker_stack.py` no atrapa toda clase de bug de la Fase 8 —
  puntualmente, no hubiera atrapado el bug de `a63c4bd` por sí solo (el
  `FileNotFoundError` solo ocurre al navegar a la página de Streamlit desde
  un browser real, no al pegarle a `/_stcore/health`) — por eso
  `test_dashboard_data_sources.py` existe aparte, sin depender de Docker, y
  sí lo cubre directo.
