import pandas as pd

from churn_mlops.config import load_yaml_config
from churn_mlops.features.build_features import (
    add_engineered_features,
    build_feature_table,
    drop_excluded_columns,
    encode_target,
    split_train_test,
)

FEATURE_CONFIG = load_yaml_config("configs/features.yaml")
DATA_CONFIG = load_yaml_config("configs/data.yaml")


def _sample_raw_df(n: int = 10) -> pd.DataFrame:
    service_values = {
        col: ["Yes", "No"] * (n // 2) for col in FEATURE_CONFIG["service_columns"]
    }
    return pd.DataFrame(
        {
            "Customer ID": [f"{i:04d}-ABCDE" for i in range(n)],
            "Churn Score": [50] * n,
            "Customer Status": ["Stayed"] * n,
            "Churn Category": [None] * n,
            "Churn Reason": [None] * n,
            "Latitude": [34.0] * n,
            "Longitude": [-118.0] * n,
            "Zip Code": [90001] * n,
            "CLTV": [4500] * n,
            "Country": ["United States"] * n,
            "State": ["California"] * n,
            "Quarter": ["Q3"] * n,
            "Contract": ["Month-to-Month"] * n,
            "Tenure in Months": list(range(1, n + 1)),
            "Churn Label": (["Yes", "No"] * (n // 2))[:n],
            **service_values,
        }
    )


def test_drop_excluded_columns_removes_leakage_low_signal_and_constant() -> None:
    df = _sample_raw_df()
    result = drop_excluded_columns(df, FEATURE_CONFIG)

    excluded = (
        FEATURE_CONFIG["id_columns"]
        + FEATURE_CONFIG["leakage_columns"]
        + FEATURE_CONFIG["low_signal_columns"]
        + FEATURE_CONFIG["constant_columns"]
    )
    for col in excluded:
        assert col not in result.columns
    assert "Contract" in result.columns


def test_add_engineered_features_is_new_customer_boundary() -> None:
    threshold = FEATURE_CONFIG["new_customer_tenure_threshold"]
    df = pd.DataFrame(
        {
            "Tenure in Months": [threshold - 1, threshold],
            **{col: ["No", "No"] for col in FEATURE_CONFIG["service_columns"]},
        }
    )
    result = add_engineered_features(df, FEATURE_CONFIG)

    assert result.loc[0, "is_new_customer"] == 1
    assert result.loc[1, "is_new_customer"] == 0


def test_add_engineered_features_counts_services_correctly() -> None:
    service_columns = FEATURE_CONFIG["service_columns"]
    row = {col: "No" for col in service_columns}
    for col in service_columns[:3]:
        row[col] = "Yes"
    df = pd.DataFrame([{"Tenure in Months": 10, **row}])

    result = add_engineered_features(df, FEATURE_CONFIG)

    assert result.loc[0, "num_extra_services"] == 3


def test_encode_target_maps_yes_no_to_binary_int() -> None:
    df = pd.DataFrame({"Churn Label": ["Yes", "No", "No"]})
    y = encode_target(df, "Churn Label")

    assert y.tolist() == [1, 0, 0]
    assert y.dtype.kind == "i"


def test_build_feature_table_excludes_leakage_and_target_from_x() -> None:
    df = _sample_raw_df(n=10)
    X, y = build_feature_table(df, FEATURE_CONFIG, DATA_CONFIG["target_column"])

    for col in FEATURE_CONFIG["leakage_columns"]:
        assert col not in X.columns
    assert DATA_CONFIG["target_column"] not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_split_train_test_is_stratified() -> None:
    n = 50
    X = pd.DataFrame({"feature": range(n)})
    y = pd.Series(([1] * 15 + [0] * 35), name="Churn Label")

    X_train, X_test, y_train, y_test = split_train_test(X, y, DATA_CONFIG)

    expected_test_size = round(n * DATA_CONFIG["test_size"])
    assert len(X_test) == expected_test_size
    assert len(X_train) + len(X_test) == n

    overall_rate = y.mean()
    assert abs(y_train.mean() - overall_rate) < 0.05
    assert abs(y_test.mean() - overall_rate) < 0.05
