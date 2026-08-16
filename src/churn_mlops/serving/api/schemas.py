"""Schemas Pydantic de la API (Fase 7) — validan objetos individuales de
request/response. Distinto del schema Pandera de `data/schema.py`, que valida
DataFrames de batches crudos (límite de capas ya establecido en CLAUDE.md).

`CustomerFeatures` espera los mismos 37 campos crudos que
`features/build_features.py::add_engineered_features` necesita para derivar
`is_new_customer`/`num_extra_services` — son las columnas de `train.parquet`/
`test.parquet` menos esas dos derivadas y el target. Los nombres de columna
originales tienen espacios (no son identificadores Python válidos), por eso
cada campo usa un alias explícito con el nombre exacto; `model_dump(by_alias=True)`
reconstruye esos nombres para armar el DataFrame que consume el Pipeline.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """Atributos crudos de un cliente, tal como los describe `data/schema.py`
    (menos las columnas de leakage/low-signal/constantes/id que
    `features/build_features.py::drop_excluded_columns` ya excluye antes de
    llegar a este punto)."""

    model_config = ConfigDict(populate_by_name=True)

    gender: Literal["Male", "Female"] = Field(alias="Gender")
    age: int = Field(alias="Age", ge=0, le=120)
    under_30: _YesNo = Field(alias="Under 30")
    senior_citizen: _YesNo = Field(alias="Senior Citizen")
    married: _YesNo = Field(alias="Married")
    dependents: _YesNo = Field(alias="Dependents")
    number_of_dependents: int = Field(alias="Number of Dependents", ge=0)
    city: str = Field(alias="City")
    population: int = Field(alias="Population", ge=0)
    referred_a_friend: _YesNo = Field(alias="Referred a Friend")
    number_of_referrals: int = Field(alias="Number of Referrals", ge=0)
    tenure_in_months: int = Field(alias="Tenure in Months", ge=0)
    offer: Literal["Offer A", "Offer B", "Offer C", "Offer D", "Offer E"] | None = (
        Field(default=None, alias="Offer")
    )
    phone_service: _YesNo = Field(alias="Phone Service")
    avg_monthly_long_distance_charges: float = Field(
        alias="Avg Monthly Long Distance Charges", ge=0
    )
    multiple_lines: _YesNo = Field(alias="Multiple Lines")
    internet_service: _YesNo = Field(alias="Internet Service")
    internet_type: Literal["Cable", "DSL", "Fiber Optic"] | None = Field(
        default=None, alias="Internet Type"
    )
    avg_monthly_gb_download: int = Field(alias="Avg Monthly GB Download", ge=0)
    online_security: _YesNo = Field(alias="Online Security")
    online_backup: _YesNo = Field(alias="Online Backup")
    device_protection_plan: _YesNo = Field(alias="Device Protection Plan")
    premium_tech_support: _YesNo = Field(alias="Premium Tech Support")
    streaming_tv: _YesNo = Field(alias="Streaming TV")
    streaming_movies: _YesNo = Field(alias="Streaming Movies")
    streaming_music: _YesNo = Field(alias="Streaming Music")
    unlimited_data: _YesNo = Field(alias="Unlimited Data")
    contract: Literal["Month-to-Month", "One Year", "Two Year"] = Field(
        alias="Contract"
    )
    paperless_billing: _YesNo = Field(alias="Paperless Billing")
    payment_method: Literal["Bank Withdrawal", "Credit Card", "Mailed Check"] = Field(
        alias="Payment Method"
    )
    monthly_charge: float = Field(alias="Monthly Charge")
    total_charges: float = Field(alias="Total Charges", ge=0)
    total_refunds: float = Field(alias="Total Refunds", ge=0)
    total_extra_data_charges: int = Field(alias="Total Extra Data Charges", ge=0)
    total_long_distance_charges: float = Field(
        alias="Total Long Distance Charges", ge=0
    )
    total_revenue: float = Field(alias="Total Revenue", ge=0)
    satisfaction_score: int = Field(alias="Satisfaction Score", ge=1, le=5)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    champion_version: str


class ExplanationResponse(PredictionResponse):
    """Extiende la predicción con SHAP values por feature transformada
    (nombres de `ColumnTransformer.get_feature_names_out()`) y el valor base
    del explainer (log-odds esperado antes de ver ninguna feature)."""

    shap_values: dict[str, float]
    base_value: float


class ModelInfoResponse(BaseModel):
    registered_model_name: str
    champion_alias: str
    champion_version: str | None
