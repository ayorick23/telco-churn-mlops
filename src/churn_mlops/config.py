from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Lee un archivo de configuración YAML de configs/ y lo devuelve como dict."""
    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
