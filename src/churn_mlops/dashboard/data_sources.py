"""Fuentes de datos auxiliares del dashboard que no son reportes estáticos.

`load_known_cities` alimenta el selector de City del formulario de
predicción: dejar ese campo como texto libre permitiría que el usuario
escriba una ciudad inexistente en el dataset — el champion actual la ignora
vía `remainder='drop'` (ADR 0009), pero una futura familia sí podría usarla
nativa (CatBoost, mismo ADR), así que el contrato de la API se mantiene
honesto restringiendo a ciudades reales.

Lee reports/known_cities.json (precomputado por
`export_known_cities.py`), no train.parquet directo: el dashboard no
empaqueta data/ en su imagen Docker (cliente HTTP puro, ver ADR 0013)."""

import json
from pathlib import Path


def load_known_cities(cities_path: str = "reports/known_cities.json") -> list[str]:
    cities: list[str] = json.loads(Path(cities_path).read_text(encoding="utf-8"))
    return cities
