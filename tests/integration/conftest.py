"""Mismo payload que tests/unit/serving/api/conftest.py — duplicado a
propósito en vez de compartido entre directorios: pytest no comparte
conftest.py entre carpetas hermanas sin un conftest.py común en la raíz de
tests/, y no vale la pena introducir uno solo para esto."""

import pytest


@pytest.fixture
def valid_customer_payload() -> dict[str, object]:
    return {
        "Gender": "Male",
        "Age": 74,
        "Under 30": "No",
        "Senior Citizen": "Yes",
        "Married": "No",
        "Dependents": "No",
        "Number of Dependents": 0,
        "City": "Westlake Village",
        "Population": 18735,
        "Referred a Friend": "No",
        "Number of Referrals": 0,
        "Tenure in Months": 43,
        "Offer": None,
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": 8.81,
        "Multiple Lines": "Yes",
        "Internet Service": "Yes",
        "Internet Type": "Fiber Optic",
        "Avg Monthly GB Download": 7,
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": "Two Year",
        "Paperless Billing": "Yes",
        "Payment Method": "Bank Withdrawal",
        "Monthly Charge": 84.85,
        "Total Charges": 3645.6,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 0,
        "Total Long Distance Charges": 378.83,
        "Total Revenue": 4024.43,
        "Satisfaction Score": 4,
    }
