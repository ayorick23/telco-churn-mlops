import json
from pathlib import Path

import pandas as pd

from churn_mlops.dashboard.export_known_cities import (
    extract_known_cities,
    save_known_cities,
)


def test_extract_known_cities_returns_sorted_unique_values() -> None:
    train_df = pd.DataFrame({"City": ["Zion", "Acton", "Acton", "Bell"]})

    result = extract_known_cities(train_df)

    assert result == ["Acton", "Bell", "Zion"]


def test_save_known_cities_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "known_cities.json"

    result_path = save_known_cities(["Acton", "Bell"], output_path)

    assert result_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == ["Acton", "Bell"]
