import json
from pathlib import Path

from churn_mlops.dashboard.data_sources import load_known_cities


def test_load_known_cities_reads_precomputed_json(tmp_path: Path) -> None:
    cities_path = tmp_path / "known_cities.json"
    cities_path.write_text(json.dumps(["Acton", "Bell", "Zion"]), encoding="utf-8")

    result = load_known_cities(str(cities_path))

    assert result == ["Acton", "Bell", "Zion"]
