from pathlib import Path

import pandas as pd

from churn_mlops.training.dataset import load_processed_datasets


def test_load_processed_datasets_splits_features_and_target(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    train_df = pd.DataFrame(
        {"Age": [30, 40, 50], "Contract": ["A", "B", "C"], "Churn Label": [0, 1, 0]}
    )
    test_df = pd.DataFrame({"Age": [60], "Contract": ["D"], "Churn Label": [1]})
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    data_config = {"processed_dir": str(processed_dir), "target_column": "Churn Label"}

    X_train, X_test, y_train, y_test = load_processed_datasets(data_config)

    assert "Churn Label" not in X_train.columns
    assert "Churn Label" not in X_test.columns
    assert len(X_train) == len(y_train) == 3
    assert len(X_test) == len(y_test) == 1
    assert y_test.tolist() == [1]
