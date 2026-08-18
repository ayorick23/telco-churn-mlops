"""Precomputa la lista de ciudades reales de train.parquet como un artefacto
liviano (reports/known_cities.json) que dashboard/data_sources.py puede leer
sin necesitar el parquet completo. Existe porque data/processed/ no se copia
a docker/dashboard.Dockerfile (el dashboard es cliente HTTP puro, ver ADR
0013) — este JSON sí viaja con reports/, igual que los demás reportes
estáticos de Producción/Drift. Se corre a mano cuando train.parquet cambia
(no es stage de dvc.yaml, mismo patrón que run_monitoring.py):

    uv run python -m churn_mlops.dashboard.export_known_cities
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from churn_mlops.config import load_yaml_config


def extract_known_cities(train_df: pd.DataFrame) -> list[str]:
    return sorted(train_df["City"].unique().tolist())


def save_known_cities(cities: list[str], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cities, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def main(data_config: dict[str, Any] | None = None) -> None:
    data_config = data_config or load_yaml_config("configs/data.yaml")
    train_df = pd.read_parquet(
        Path(data_config["processed_dir"]) / "train.parquet", columns=["City"]
    )
    cities = extract_known_cities(train_df)
    output_path = save_known_cities(cities, "reports/known_cities.json")
    print(f"{len(cities)} ciudades guardadas en {output_path}")


if __name__ == "__main__":
    main()
