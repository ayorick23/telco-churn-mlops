"""Fuentes de datos auxiliares del dashboard que no son reportes estáticos.

`load_known_cities` alimenta el selector de City del formulario de
predicción: dejar ese campo como texto libre permitiría que el usuario
escriba una ciudad inexistente en el dataset — el champion actual la ignora
vía `remainder='drop'` (ADR 0009), pero una futura familia sí podría usarla
nativa (CatBoost, mismo ADR), así que el contrato de la API se mantiene
honesto restringiendo a ciudades reales."""

from pathlib import Path

import pandas as pd

from churn_mlops.config import load_yaml_config


def load_known_cities(data_config_path: str = "configs/data.yaml") -> list[str]:
    data_config = load_yaml_config(data_config_path)
    train_path = Path(data_config["processed_dir"]) / "train.parquet"
    cities = pd.read_parquet(train_path, columns=["City"])["City"]
    return sorted(cities.unique().tolist())
