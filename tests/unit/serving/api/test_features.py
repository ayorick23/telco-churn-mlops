from churn_mlops.serving.api.features import customers_to_model_input
from churn_mlops.serving.api.schemas import CustomerFeatures


def test_customers_to_model_input_adds_engineered_columns(
    valid_customer_payload: dict[str, object],
) -> None:
    customer = CustomerFeatures(**valid_customer_payload)

    X = customers_to_model_input([customer])

    assert "is_new_customer" in X.columns
    assert "num_extra_services" in X.columns
    # Tenure in Months=43 >= threshold (12, configs/features.yaml) -> no es nuevo.
    assert X.loc[0, "is_new_customer"] == 0


def test_customers_to_model_input_counts_extra_services(
    valid_customer_payload: dict[str, object],
) -> None:
    payload = {
        **valid_customer_payload,
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Streaming Music": "No",
        "Unlimited Data": "No",
        "Multiple Lines": "No",
    }
    customer = CustomerFeatures(**payload)

    X = customers_to_model_input([customer])

    assert X.loc[0, "num_extra_services"] == 2


def test_customers_to_model_input_handles_multiple_rows(
    valid_customer_payload: dict[str, object],
) -> None:
    customers = [CustomerFeatures(**valid_customer_payload) for _ in range(3)]

    X = customers_to_model_input(customers)

    assert len(X) == 3
