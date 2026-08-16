"""Convierte requests de la API (`CustomerFeatures`) al DataFrame de 39
columnas que espera el Pipeline de producción. Reusa
`features/build_features.py::add_engineered_features` (Fase 2) en vez de
reimplementar el cálculo de `is_new_customer`/`num_extra_services` — mismo
`configs/features.yaml` que usa el pipeline de DVC."""

import pandas as pd

from churn_mlops.config import load_yaml_config
from churn_mlops.features.build_features import add_engineered_features
from churn_mlops.serving.api.schemas import CustomerFeatures

_FEATURE_CONFIG_PATH = "configs/features.yaml"


def customers_to_model_input(customers: list[CustomerFeatures]) -> pd.DataFrame:
    """Una o más `CustomerFeatures` -> DataFrame con las 39 columnas del
    Pipeline (37 crudas + is_new_customer + num_extra_services)."""
    feature_config = load_yaml_config(_FEATURE_CONFIG_PATH)
    raw_df = pd.DataFrame([c.model_dump(by_alias=True) for c in customers])
    return add_engineered_features(raw_df, feature_config)
