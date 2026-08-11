"""Arma el Pipeline sin fitear de cada familia. Cada framework encodea categóricas a
su manera (ADR 0007, ADR 0009): LR/XGBoost/LightGBM usan ColumnTransformer con
one-hot + (scaling solo para LR); CatBoost usa categóricas nativas vía cat_features."""

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from churn_mlops.training.preprocessing import (
    build_onehot_preprocessor,
    fill_categorical_missing,
    infer_column_types,
)


def _onehot_columns(
    X_train: pd.DataFrame, high_cardinality_columns: list[str]
) -> tuple[list[str], list[str]]:
    numeric_columns, categorical_columns = infer_column_types(X_train)
    categorical_columns = [
        c for c in categorical_columns if c not in high_cardinality_columns
    ]
    return numeric_columns, categorical_columns


def build_logistic_regression_pipeline(
    X_train: pd.DataFrame, high_cardinality_columns: list[str], params: dict[str, Any]
) -> Pipeline:
    """LR: one-hot + scaling; dropea columnas de alta cardinalidad."""
    numeric_columns, categorical_columns = _onehot_columns(
        X_train, high_cardinality_columns
    )
    preprocessor = build_onehot_preprocessor(
        numeric_columns, categorical_columns, scale_numeric=True
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", LogisticRegression(**params))]
    )


def build_xgboost_pipeline(
    X_train: pd.DataFrame, high_cardinality_columns: list[str], params: dict[str, Any]
) -> Pipeline:
    """XGBoost: one-hot, sin scaling; dropea columnas de alta cardinalidad."""
    numeric_columns, categorical_columns = _onehot_columns(
        X_train, high_cardinality_columns
    )
    preprocessor = build_onehot_preprocessor(
        numeric_columns, categorical_columns, scale_numeric=False
    )
    return Pipeline([("preprocess", preprocessor), ("model", XGBClassifier(**params))])


def build_lightgbm_pipeline(
    X_train: pd.DataFrame, high_cardinality_columns: list[str], params: dict[str, Any]
) -> Pipeline:
    """LightGBM: one-hot, sin scaling; dropea columnas de alta cardinalidad."""
    numeric_columns, categorical_columns = _onehot_columns(
        X_train, high_cardinality_columns
    )
    preprocessor = build_onehot_preprocessor(
        numeric_columns, categorical_columns, scale_numeric=False
    )
    return Pipeline([("preprocess", preprocessor), ("model", LGBMClassifier(**params))])


def build_catboost_pipeline(X_train: pd.DataFrame, params: dict[str, Any]) -> Pipeline:
    """CatBoost: categóricas nativas vía cat_features (incluye City) — ADR 0009.
    Solo rellena NaN estructural en categóricas; numéricas las maneja CatBoost."""
    _, categorical_columns = infer_column_types(X_train)
    fill_missing = FunctionTransformer(
        fill_categorical_missing, kw_args={"categorical_columns": categorical_columns}
    )
    model = CatBoostClassifier(cat_features=categorical_columns, **params)
    return Pipeline([("preprocess", fill_missing), ("model", model)])
