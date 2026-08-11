"""Preprocessing compartido por las 4 familias (ADR 0007, ADR 0009). Funciones puras:
sin fit sobre test, sin dependencias de MLflow."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa columnas por dtype. Devuelve (numeric_columns, categorical_columns)."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(include="object").columns.tolist()
    return numeric_columns, categorical_columns


def build_onehot_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Imputación + one-hot para categóricas (+ scaling opcional de numéricas).
    Columnas no listadas (ej. City) se dropean vía remainder='drop'."""
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def fill_categorical_missing(
    X: pd.DataFrame, categorical_columns: list[str], placeholder: str = "Missing"
) -> pd.DataFrame:
    """Rellena NaN estructural (Offer, Internet Type) con texto: CatBoost no acepta
    NaN float en columnas categóricas nativas."""
    X = X.copy()
    X[categorical_columns] = X[categorical_columns].fillna(placeholder)
    return X


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio negativo/positivo para XGBoost (sin class_weight='balanced' nativo)."""
    counts = y_train.value_counts()
    return float(counts.loc[0] / counts.loc[1])
