"""Feature engineering puro: sin lógica de modelo (eso vive en training/, Fase 3).

Aplica las exclusiones de configs/features.yaml (leakage, bajo valor, columnas
constantes/identificadoras), agrega features derivadas, codifica el target y
genera el split train/test versionado que consume Fase 3.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_mlops.config import load_yaml_config
from churn_mlops.data.load import load_raw_data


def drop_excluded_columns(
    df: pd.DataFrame, feature_config: dict[str, Any]
) -> pd.DataFrame:
    """Dropea id/leakage/low-signal/constant columns definidas en configs/features.yaml."""
    excluded = (
        feature_config["id_columns"]
        + feature_config["leakage_columns"]
        + feature_config["low_signal_columns"]
        + feature_config["constant_columns"]
    )
    result: pd.DataFrame = df.drop(columns=excluded)
    return result


def add_engineered_features(
    df: pd.DataFrame, feature_config: dict[str, Any]
) -> pd.DataFrame:
    """Agrega is_new_customer y num_extra_services (EDA sección 8)."""
    df = df.copy()

    threshold = feature_config["new_customer_tenure_threshold"]
    df["is_new_customer"] = (df["Tenure in Months"] < threshold).astype(int)

    service_columns = feature_config["service_columns"]
    df["num_extra_services"] = (df[service_columns] == "Yes").sum(axis=1)

    return df


def encode_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Mapea el target Yes/No a 0/1. Es universal para cualquier modelo, a diferencia
    del encoding de features (que vive en training/), por eso se hace acá."""
    return (df[target_column] == "Yes").astype(int).rename(target_column)


def build_feature_table(
    df: pd.DataFrame, feature_config: dict[str, Any], target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Orquesta drop + engineering + encoding del target. Devuelve (X, y)."""
    df = drop_excluded_columns(df, feature_config)
    df = add_engineered_features(df, feature_config)
    y = encode_target(df, target_column)
    X = df.drop(columns=[target_column])
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, data_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado, con random_state fijo desde configs/data.yaml."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_column: str,
    output_dir: str | Path,
) -> None:
    """Escribe train.parquet y test.parquet (features + target) en output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[target_column] = y_train
    test_df = X_test.copy()
    test_df[target_column] = y_test

    train_df.to_parquet(output_path / "train.parquet", index=False)
    test_df.to_parquet(output_path / "test.parquet", index=False)


def main() -> None:
    data_config = load_yaml_config("configs/data.yaml")
    feature_config = load_yaml_config("configs/features.yaml")
    target_column = data_config["target_column"]

    df = load_raw_data(data_config["raw_path"])
    X, y = build_feature_table(df, feature_config, target_column)
    X_train, X_test, y_train, y_test = split_train_test(X, y, data_config)
    save_processed(
        X_train,
        X_test,
        y_train,
        y_test,
        target_column=target_column,
        output_dir=data_config["processed_dir"],
    )


if __name__ == "__main__":
    main()
