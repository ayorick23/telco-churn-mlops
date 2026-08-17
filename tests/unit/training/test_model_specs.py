import numpy as np
import pandas as pd
import pytest

from churn_mlops.training.model_specs import (
    build_catboost_pipeline,
    build_lightgbm_pipeline,
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
)

HIGH_CARDINALITY_COLUMNS = ["City"]


def _sample_data(n: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    cities = [f"City-{i}" for i in range(20)]
    contract = rng.choice(["Month-to-Month", "One Year", "Two Year"], size=n)
    offer = rng.choice(["Offer A", "Offer B", None], size=n)
    X = pd.DataFrame(
        {
            "Age": rng.integers(18, 80, size=n),
            "Monthly Charge": rng.uniform(20, 120, size=n),
            "Contract": contract,
            "Offer": offer,
            "City": rng.choice(cities, size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="Churn Label")
    # Asegura ambas clases presentes.
    y.iloc[0] = 0
    y.iloc[1] = 1
    return X, y


@pytest.mark.parametrize(
    "builder",
    [
        build_logistic_regression_pipeline,
        build_xgboost_pipeline,
        build_lightgbm_pipeline,
    ],
)
def test_onehot_pipelines_fit_predict_and_drop_city(builder) -> None:  # type: ignore[no-untyped-def]
    X, y = _sample_data()
    pipeline = builder(X, HIGH_CARDINALITY_COLUMNS, {})
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(predictions.tolist()) <= {0, 1}
    assert probabilities.shape == (len(X), 2)

    categorical_columns_used = pipeline.named_steps["preprocess"].transformers[1][2]
    assert "City" not in categorical_columns_used


def test_catboost_pipeline_fits_with_city_native_and_handles_missing() -> None:
    X, y = _sample_data()
    pipeline = build_catboost_pipeline(X, {"iterations": 10, "verbose": False})
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), 2)

    cat_features = pipeline.named_steps["model"].get_params()["cat_features"]
    assert "City" in cat_features
