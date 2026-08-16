from pathlib import Path

import pandas as pd
import yaml
from churn_mlops.dashboard.data_sources import load_known_cities


def test_load_known_cities_returns_sorted_unique_values(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame({"City": ["Zion", "Acton", "Acton", "Bell"]}).to_parquet(
        processed_dir / "train.parquet"
    )
    data_config_path = tmp_path / "data.yaml"
    data_config_path.write_text(
        yaml.dump({"processed_dir": str(processed_dir)}), encoding="utf-8"
    )

    result = load_known_cities(str(data_config_path))

    assert result == ["Acton", "Bell", "Zion"]
