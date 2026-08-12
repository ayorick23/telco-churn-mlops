"""Tuning de LightGBM con Optuna: Stratified K-Fold sobre train.parquet, optimiza
el F1 promedio de la clase positiva entre folds (ADR 0010). test.parquet nunca se
toca acá — se evalúa una sola vez, en run_training.py, con los mejores params."""

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from churn_mlops.training.evaluation import compute_classification_metrics
from churn_mlops.training.model_specs import build_lightgbm_pipeline


def build_search_space(
    trial: optuna.Trial, search_space_config: dict[str, Any]
) -> dict[str, Any]:
    """Arma los hiperparámetros sugeridos por un trial a partir del espacio
    declarado en configs/training.yaml:optuna.search_space."""
    params: dict[str, Any] = {}
    for name, spec in search_space_config.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        else:
            raise ValueError(f"Tipo de hiperparámetro desconocido: {spec['type']}")
    return params


def cross_validate_lightgbm(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    high_cardinality_columns: list[str],
    cv_folds: int,
    random_state: int,
) -> float:
    """Arma y fitea un Pipeline nuevo por fold (evita leakage, ADR 0007): el
    ColumnTransformer se fitea únicamente sobre la porción train del fold, nunca
    sobre la porción val. Devuelve el F1 promedio de la clase positiva entre
    folds."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_f1_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline = build_lightgbm_pipeline(
            X_fold_train, high_cardinality_columns, params
        )
        pipeline.fit(X_fold_train, y_fold_train)
        y_pred = pipeline.predict(X_fold_val)
        y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
        metrics = compute_classification_metrics(y_fold_val.to_numpy(), y_pred, y_proba)
        fold_f1_scores.append(metrics["f1"])

    return float(np.mean(fold_f1_scores))


def make_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    high_cardinality_columns: list[str],
    training_config: dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Cierra sobre train.parquet y devuelve la función objetivo que Optuna
    maximiza: F1 promedio de cross_validate_lightgbm para los params del trial."""
    optuna_config = training_config["optuna"]
    fixed_params = training_config["fixed_params"]

    def objective(trial: optuna.Trial) -> float:
        suggested_params = build_search_space(trial, optuna_config["search_space"])
        params = {**fixed_params, **suggested_params}
        return cross_validate_lightgbm(
            params,
            X_train,
            y_train,
            high_cardinality_columns,
            optuna_config["cv_folds"],
            fixed_params["random_state"],
        )

    return objective


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    high_cardinality_columns: list[str],
    training_config: dict[str, Any],
) -> optuna.Study:
    """Corre el estudio de Optuna (TPESampler con seed fija, para reproducibilidad)
    y devuelve el study completo — best_params/best_value vía
    study.best_params/study.best_value."""
    optuna_config = training_config["optuna"]
    fixed_params = training_config["fixed_params"]

    sampler = optuna.samplers.TPESampler(seed=fixed_params["random_state"])
    study = optuna.create_study(direction=optuna_config["direction"], sampler=sampler)
    objective = make_objective(
        X_train, y_train, high_cardinality_columns, training_config
    )
    study.optimize(
        objective, n_trials=optuna_config["n_trials"], show_progress_bar=False
    )
    return study
