"""Schema Pandera del CSV crudo de Telco Customer Churn (7043 filas x 50 columnas).

Valores categóricos y nulls estructurales confirmados en notebooks/01_eda.ipynb.
"""

import pandera.pandas as pa

_YES_NO_COLUMNS = [
    "Under 30",
    "Senior Citizen",
    "Married",
    "Dependents",
    "Referred a Friend",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Paperless Billing",
    "Churn Label",
]

_yes_no_columns = {
    name: pa.Column(str, pa.Check.isin(["Yes", "No"])) for name in _YES_NO_COLUMNS
}

raw_telco_schema = pa.DataFrameSchema(
    {
        "Customer ID": pa.Column(str, unique=True),
        "Gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "Age": pa.Column(int, pa.Check.in_range(0, 120)),
        **_yes_no_columns,
        "Number of Dependents": pa.Column(int, pa.Check.ge(0)),
        "Country": pa.Column(str),
        "State": pa.Column(str),
        "City": pa.Column(str),
        "Zip Code": pa.Column(int),
        "Latitude": pa.Column(float),
        "Longitude": pa.Column(float),
        "Population": pa.Column(int, pa.Check.ge(0)),
        "Quarter": pa.Column(str),
        "Number of Referrals": pa.Column(int, pa.Check.ge(0)),
        "Tenure in Months": pa.Column(int, pa.Check.ge(0)),
        # Nullable: solo aplica a clientes con una oferta promocional activa.
        "Offer": pa.Column(
            str,
            pa.Check.isin(["Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]),
            nullable=True,
        ),
        "Avg Monthly Long Distance Charges": pa.Column(float, pa.Check.ge(0)),
        # Nullable: solo aplica a clientes con Internet Service = Yes.
        "Internet Type": pa.Column(
            str, pa.Check.isin(["Cable", "DSL", "Fiber Optic"]), nullable=True
        ),
        "Avg Monthly GB Download": pa.Column(int, pa.Check.ge(0)),
        "Contract": pa.Column(
            str, pa.Check.isin(["Month-to-Month", "One Year", "Two Year"])
        ),
        "Payment Method": pa.Column(
            str, pa.Check.isin(["Bank Withdrawal", "Credit Card", "Mailed Check"])
        ),
        "Monthly Charge": pa.Column(float),
        "Total Charges": pa.Column(float, pa.Check.ge(0)),
        "Total Refunds": pa.Column(float, pa.Check.ge(0)),
        "Total Extra Data Charges": pa.Column(int, pa.Check.ge(0)),
        "Total Long Distance Charges": pa.Column(float, pa.Check.ge(0)),
        "Total Revenue": pa.Column(float, pa.Check.ge(0)),
        "Satisfaction Score": pa.Column(int, pa.Check.in_range(1, 5)),
        "Customer Status": pa.Column(
            str, pa.Check.isin(["Churned", "Joined", "Stayed"])
        ),
        "Churn Score": pa.Column(int, pa.Check.in_range(0, 100)),
        "CLTV": pa.Column(int, pa.Check.ge(0)),
        # Nullable: solo aplica a clientes con Churn Label = Yes.
        "Churn Category": pa.Column(
            str,
            pa.Check.isin(
                ["Attitude", "Competitor", "Dissatisfaction", "Other", "Price"]
            ),
            nullable=True,
        ),
        # Nullable: solo aplica a clientes con Churn Label = Yes. Alta cardinalidad
        # (20 valores distintos), no se restringe a un set fijo.
        "Churn Reason": pa.Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
)
