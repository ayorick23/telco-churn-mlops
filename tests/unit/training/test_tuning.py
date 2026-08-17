import numpy as np
import optuna
import pandas as pd

from churn_mlops.training.tuning import (
    build_search_space,
    cross_validate_lightgbm,
    run_optuna_search,
)

HIGH_CARDINALITY_COLUMNS = ["City"]

SMALL_SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 10, "high": 50, "step": 10},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
}


def _sample_data(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    cities = [f"City-{i}" for i in range(20)]
    X = pd.DataFrame(
        {
            "Age": rng.integers(18, 80, size=n),
            "Monthly Charge": rng.uniform(20, 120, size=n),
            "Contract": rng.choice(["Month-to-Month", "One Year", "Two Year"], size=n),
            "Offer": rng.choice(["Offer A", "Offer B", None], size=n),
            "City": rng.choice(cities, size=n),
        }
    )
    # Clases balanceadas por construcción: StratifiedKFold necesita ambas clases
    # presentes en cada fold, no solo en el dataset completo.
    y = pd.Series([0, 1] * (n // 2), name="Churn Label")
    return X, y


def test_build_search_space_respects_bounds() -> None:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    params = build_search_space(trial, SMALL_SEARCH_SPACE)

    assert 10 <= params["n_estimators"] <= 50
    assert (params["n_estimators"] - 10) % 10 == 0
    assert 0.05 <= params["learning_rate"] <= 0.3
    study.tell(trial, 0.0)


def test_cross_validate_lightgbm_returns_f1_in_valid_range() -> None:
    X, y = _sample_data()
    params = {
        "n_estimators": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
    }

    score = cross_validate_lightgbm(
        params, X, y, HIGH_CARDINALITY_COLUMNS, cv_folds=2, random_state=42
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_run_optuna_search_returns_study_with_best_params_from_search_space() -> None:
    X, y = _sample_data()
    training_config = {
        "optuna": {
            "n_trials": 2,
            "cv_folds": 2,
            "direction": "maximize",
            "search_space": SMALL_SEARCH_SPACE,
        },
        "fixed_params": {"class_weight": "balanced", "random_state": 42, "verbose": -1},
    }

    study = run_optuna_search(X, y, HIGH_CARDINALITY_COLUMNS, training_config)

    assert set(study.best_params.keys()) == set(SMALL_SEARCH_SPACE.keys())
    assert 0.0 <= study.best_value <= 1.0


def test_run_optuna_search_is_reproducible_with_same_seed() -> None:
    X, y = _sample_data()
    training_config = {
        "optuna": {
            "n_trials": 2,
            "cv_folds": 2,
            "direction": "maximize",
            "search_space": SMALL_SEARCH_SPACE,
        },
        "fixed_params": {"class_weight": "balanced", "random_state": 42, "verbose": -1},
    }

    study_a = run_optuna_search(X, y, HIGH_CARDINALITY_COLUMNS, training_config)
    study_b = run_optuna_search(X, y, HIGH_CARDINALITY_COLUMNS, training_config)

    assert study_a.best_params == study_b.best_params
