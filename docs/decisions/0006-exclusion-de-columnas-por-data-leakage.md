# 6. Exclusión de columnas por data leakage

**Fecha:** 2026-08-08
**Estado:** Aceptada
**Fase:** 1 — Setup y EDA (EDA, `notebooks/01_eda.ipynb`, sección 7)

## Contexto

Durante el EDA se identificaron columnas del dataset que están calculadas *a partir* del churn, o que solo existen *después* de que el churn ya ocurrió:

| Columna | Motivo |
|---|---|
| `Churn Score` | Score de propensión a churn calculado por la empresa — ya es casi la predicción. |
| `Customer Status` | Incluye directamente el valor "Churned" — es el target codificado de otra forma. |
| `Churn Category` | Solo tiene valor para clientes que ya se dieron de baja (1 869 de 7 043 filas). |
| `Churn Reason` | Igual que `Churn Category`: solo existe post-churn. |

Usarlas como input haría que el modelo aprenda a copiar la respuesta en vez de predecirla: funcionaría perfecto en entrenamiento y fallaría en producción, donde esos datos no existen todavía para un cliente activo.

## Decisión

Excluir estas cuatro columnas del feature set en Fase 2. `Churn Label` es el único target válido para el modelo.

## Alternativas consideradas

Ninguna — es leakage clásico, no hay escenario de uso real donde estas columnas estén disponibles al momento de predecir.

## Consecuencias

- La capa `data`/`features` de Fase 2 debe dropear estas columnas explícitamente, o el schema de Pandera debe excluirlas del set de features validado.
- Cualquier feature engineering que las use debe rechazarse en code review.
- Otras columnas de baja señal (`Latitude`, `Longitude`, `Zip Code`, `CLTV`) se excluyen también, pero por señal/costo de encoding, no por leakage — ver sección 8 del EDA.
