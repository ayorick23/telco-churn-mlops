from churn_mlops.dashboard.shap_labels import humanize_shap_feature_name

KNOWN_COLUMNS = ["Age", "Contract", "Payment Method", "Streaming TV", "Streaming Music"]


def test_humanizes_numeric_prefix_to_bare_column_name() -> None:
    result = humanize_shap_feature_name("numeric__Age", KNOWN_COLUMNS)

    assert result == "Age"


def test_humanizes_categorical_prefix_to_column_colon_category() -> None:
    result = humanize_shap_feature_name(
        "categorical__Contract_Month-to-Month", KNOWN_COLUMNS
    )

    assert result == "Contract: Month-to-Month"


def test_handles_column_and_category_names_with_spaces() -> None:
    result = humanize_shap_feature_name(
        "categorical__Payment Method_Bank Withdrawal", KNOWN_COLUMNS
    )

    assert result == "Payment Method: Bank Withdrawal"


def test_does_not_confuse_similarly_prefixed_columns() -> None:
    result = humanize_shap_feature_name(
        "categorical__Streaming Music_Yes", KNOWN_COLUMNS
    )

    assert result == "Streaming Music: Yes"


def test_returns_name_unchanged_when_no_double_underscore() -> None:
    result = humanize_shap_feature_name("Age", KNOWN_COLUMNS)

    assert result == "Age"
