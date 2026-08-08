# 3. Calidad de código: Ruff + MyPy, split entre pre-commit y CI

**Fecha:** 2026-08-07
**Estado:** Aceptada
**Fase:** 1 — Setup y EDA

## Contexto

Se quiere una configuración de calidad de código simple, sin duplicar herramientas, y sin ralentizar el ciclo de commit local con checks pesados sobre un proyecto con dependencias de ML grandes.

## Decisión

- **Ruff** como único linter + formatter (reemplaza Flake8 + isort + Black).
- **MyPy** para type-checking, pero **solo en CI**, no en pre-commit — es lento con las dependencias de ML instaladas.
- Hooks de pre-commit: `ruff`, `ruff-format`, `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` (excluyendo `uv.lock`), `check-merge-conflict`.

## Alternativas consideradas

- **MyPy también en pre-commit** — rechazado: fricciona el flujo local en cada commit.
- **Black + isort + Flake8 por separado** — rechazado: más herramientas y configuración que Ruff, que ya cubre todo eso más rápido.

## Consecuencias

- Los commits locales son rápidos; MyPy solo se valida en GitHub Actions antes de mergear a `main`.
- `uv.lock` está explícitamente excluido del check de archivos grandes para no bloquear commits legítimos.
