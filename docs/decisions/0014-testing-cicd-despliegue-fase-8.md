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
