"""Lee el split de Fase 2 — Fase 3 nunca vuelve a splitear (ADR 0008)."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_processed_datasets(
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Lee train/test.parquet de configs/data.yaml:processed_dir. Devuelve
    (X_train, X_test, y_train, y_test)."""
    processed_dir = Path(data_config["processed_dir"])
    target_column = data_config["target_column"]

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    return X_train, X_test, y_train, y_test
