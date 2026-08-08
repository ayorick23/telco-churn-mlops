# 5. Estrategia de branching

**Fecha:** 2026-08-07
**Estado:** Aceptada
**Fase:** 1 — Setup y EDA

## Contexto

Proyecto de portafolio individual, sin equipo, pero se busca simular un flujo de trabajo profesional con historial legible y revisable por fase del roadmap.

## Decisión

- Una rama `feat/<scope>` por fase o feature del roadmap.
- Un PR por rama, mergeado a `main` al cerrar cada fase.
- Commits en formato **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

## Alternativas consideradas

- **Trabajar directo sobre `main`** — rechazado: no deja evidencia clara del progreso por fase, valioso para un portafolio.
- **Un único PR gigante al final** — rechazado: dificulta revisar el avance incremental y aumenta el riesgo de conflictos.

## Consecuencias

- Cada fase del roadmap queda documentada como un PR revisable de forma independiente.
- El historial de git refleja directamente las 8 fases descritas en `README.md` y `CLAUDE.md`.
