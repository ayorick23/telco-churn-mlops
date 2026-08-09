import pandas as pd
import pandera.errors
import pytest
from churn_mlops.data.schema import raw_telco_schema


def _valid_row() -> dict:
    return {
        "Customer ID": "0001-ABCDE",
        "Gender": "Male",
        "Age": 35,
        "Under 30": "No",
        "Senior Citizen": "No",
        "Married": "Yes",
        "Dependents": "No",
        "Number of Dependents": 0,
        "Country": "United States",
        "State": "California",
        "City": "Los Angeles",
        "Zip Code": 90001,
        "Latitude": 34.05,
        "Longitude": -118.25,
        "Population": 50000,
        "Quarter": "Q3",
        "Referred a Friend": "No",
        "Number of Referrals": 0,
        "Tenure in Months": 24,
        "Offer": None,
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": 10.5,
        "Multiple Lines": "No",
        "Internet Service": "Yes",
        "Internet Type": "Fiber Optic",
        "Avg Monthly GB Download": 20,
        "Online Security": "Yes",
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "Yes",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": "Month-to-Month",
        "Paperless Billing": "Yes",
        "Payment Method": "Credit Card",
        "Monthly Charge": 70.5,
        "Total Charges": 1690.0,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 0,
        "Total Long Distance Charges": 250.0,
        "Total Revenue": 1940.0,
        "Satisfaction Score": 3,
        "Customer Status": "Stayed",
        "Churn Label": "No",
        "Churn Score": 45,
        "CLTV": 4500,
        "Churn Category": None,
        "Churn Reason": None,
    }


def test_valid_row_passes() -> None:
    df = pd.DataFrame([_valid_row()])
    validated = raw_telco_schema.validate(df)
    assert len(validated) == 1


def test_invalid_categorical_value_fails() -> None:
    row = _valid_row()
    row["Churn Label"] = "Maybe"
    df = pd.DataFrame([row])
    with pytest.raises(pandera.errors.SchemaError):
        raw_telco_schema.validate(df)


def test_null_in_non_nullable_column_fails() -> None:
    row = _valid_row()
    row["Gender"] = None
    df = pd.DataFrame([row])
    with pytest.raises(pandera.errors.SchemaError):
        raw_telco_schema.validate(df)


def test_bad_dtype_fails() -> None:
    # Un valor no coercible a int hace que pandera falle en el paso de coerción
    # (SchemaErrors, plural) en vez de en un Check individual (SchemaError).
    row = _valid_row()
    row["Age"] = "thirty-five"
    df = pd.DataFrame([row])
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        raw_telco_schema.validate(df)


def test_nullable_columns_accept_null() -> None:
    # Offer, Internet Type, Churn Category, Churn Reason ya son None en _valid_row()
    # y test_valid_row_passes ya lo cubre; acá confirmamos explícitamente Internet
    # Type nulo (cliente sin Internet Service) sin romper el resto del schema.
    row = _valid_row()
    row["Internet Service"] = "No"
    row["Internet Type"] = None
    df = pd.DataFrame([row])
    validated = raw_telco_schema.validate(df)
    assert pd.isna(validated.loc[0, "Internet Type"])
