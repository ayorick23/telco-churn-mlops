import pytest
from churn_mlops.serving.api.schemas import CustomerFeatures
from pydantic import ValidationError


def test_accepts_raw_column_names_as_aliases(
    valid_customer_payload: dict[str, object],
) -> None:
    customer = CustomerFeatures(**valid_customer_payload)

    assert customer.contract == "Two Year"
    assert customer.age == 74


def test_model_dump_by_alias_restores_raw_column_names(
    valid_customer_payload: dict[str, object],
) -> None:
    customer = CustomerFeatures(**valid_customer_payload)

    dumped = customer.model_dump(by_alias=True)

    assert "Contract" in dumped
    assert "contract" not in dumped
    assert dumped["Contract"] == "Two Year"


def test_rejects_invalid_categorical_value(
    valid_customer_payload: dict[str, object],
) -> None:
    payload = {**valid_customer_payload, "Contract": "Not-A-Real-Contract"}

    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_offer_and_internet_type_accept_none(
    valid_customer_payload: dict[str, object],
) -> None:
    payload = {**valid_customer_payload, "Offer": None, "Internet Type": None}

    customer = CustomerFeatures(**payload)

    assert customer.offer is None
    assert customer.internet_type is None


def test_rejects_negative_values_on_nonnegative_fields(
    valid_customer_payload: dict[str, object],
) -> None:
    payload = {**valid_customer_payload, "Total Charges": -1.0}

    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)
