import numpy as np
import pandas as pd
from churn_mlops.training.preprocessing import (
    build_onehot_preprocessor,
    compute_scale_pos_weight,
    fill_categorical_missing,
    infer_column_types,
)


def test_infer_column_types_splits_numeric_and_categorical() -> None:
    df = pd.DataFrame(
        {
            "Age": [30, 40],
            "Monthly Charge": [50.0, 60.0],
            "Contract": ["Month-to-Month", "Two Year"],
            "Payment Method": ["Credit Card", "Mailed Check"],
        }
    )

    numeric_columns, categorical_columns = infer_column_types(df)

    assert set(numeric_columns) == {"Age", "Monthly Charge"}
    assert set(categorical_columns) == {"Contract", "Payment Method"}


def test_build_onehot_preprocessor_output_has_no_nan_and_drops_unlisted_columns() -> (
    None
):
    df = pd.DataFrame(
        {
            "Age": [30, 40, 50, np.nan],
            "Contract": ["Month-to-Month", "Two Year", None, "One Year"],
            "City": ["A", "B", "C", "D"],
        }
    )
    numeric_columns = ["Age"]
    categorical_columns = ["Contract"]

    preprocessor = build_onehot_preprocessor(
        numeric_columns, categorical_columns, scale_numeric=True
    )
    transformed = preprocessor.fit_transform(df)

    assert not np.isnan(transformed).any()
    # 1 columna numérica + 4 categorías de Contract (incluye "Missing" imputado) = 5
    assert transformed.shape == (4, 5)


def test_fill_categorical_missing_only_touches_listed_columns() -> None:
    df = pd.DataFrame(
        {
            "Offer": ["Offer A", None, "Offer B"],
            "Age": [30.0, np.nan, 50.0],
        }
    )

    result = fill_categorical_missing(df, categorical_columns=["Offer"])

    assert result["Offer"].tolist() == ["Offer A", "Missing", "Offer B"]
    assert np.isnan(result["Age"]).sum() == 1


def test_compute_scale_pos_weight_matches_class_ratio() -> None:
    y_train = pd.Series([0] * 70 + [1] * 30)

    weight = compute_scale_pos_weight(y_train)

    assert weight == 70 / 30
