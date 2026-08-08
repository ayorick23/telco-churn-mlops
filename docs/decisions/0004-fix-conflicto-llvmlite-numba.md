# 4. Fix del conflicto llvmlite/numba (dependencia transitiva de SHAP)

**Fecha:** 2026-08-07
**Estado:** Aceptada
**Fase:** 1 — Setup y EDA

## Contexto

SHAP depende transitivamente de `numba`, que a su vez depende de `llvmlite`. Sin fijar versiones explícitas, `uv` resuelve `llvmlite==0.36.0`, que solo soporta Python <3.10 — incompatible con Python 3.11 fijado en [0001](0001-gestion-de-dependencias-y-entorno.md).

## Decisión

Pinnear explícitamente en `pyproject.toml`:

```toml
"llvmlite>=0.41",
"numba>=0.57",
```

## Alternativas consideradas

- **Bajar a Python 3.9/3.10** — rechazado: contradice la decisión ya tomada de usar 3.11.
- **Quitar SHAP del proyecto** — rechazado: la explicabilidad por predicción individual es un requisito del serving (Fase 7).

## Consecuencias

No remover estas dos líneas de `pyproject.toml` sin volver a verificar cómo resuelve `uv` el árbol de dependencias de SHAP.
