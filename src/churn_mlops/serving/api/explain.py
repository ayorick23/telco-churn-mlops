"""Explicabilidad SHAP por predicción individual (Fase 7, endpoint
`POST /explain`). `shap.TreeExplainer` corre nativo sobre el `LGBMClassifier`
(step "model" del Pipeline) usando el algoritmo tree_path_dependent por
default — no necesita un dataset de background. Los SHAP values son en el
espacio de salida raw (log-odds) de LightGBM, no de probabilidad.

Verificado contra el champion real: para este modelo binario,
`explainer(X_transformed).values` tiene shape (n_samples, n_features) y
`base_values` un escalar por fila (no una lista por clase)."""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def explain_prediction(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[dict[str, float], float]:
    """X: una sola fila (39 columnas, ya con features engineered). Devuelve
    (shap_values por nombre de feature transformada, base_value)."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_transformed)

    shap_values = dict(zip(feature_names, explanation.values[0].tolist(), strict=True))
    base_value = float(explanation.base_values[0])
    return shap_values, base_value
