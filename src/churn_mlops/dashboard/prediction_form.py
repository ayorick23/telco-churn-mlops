"""Estado y reglas del formulario de predicción interactivo. Funciones puras
sin dependencia de `streamlit` (testeables sin un runtime corriendo) —
`app.py` solo las orquesta contra los widgets.

No usamos `st.form`: dentro de un form los widgets no reaccionan entre sí
hasta el submit, y acá necesitamos que elegir "Internet Service: No" bloquee
en vivo los campos que dependen de tener internet (mismo motivo por el que
`age` deriva `under_30` en cada rerun, no solo al enviar)."""

import random
from dataclasses import replace

from churn_mlops.dashboard.form_fields import FieldSpec, build_field_specs

# Sentinel de UI para "sin valor" en selects nullable (Offer, Internet Type).
# Los widgets de Streamlit no aceptan None como opción de un selectbox, así
# que el estado interno del formulario usa este string; to_payload() lo
# traduce a None real recién al armar el request de la API.
NONE_SENTINEL = "(ninguno)"

# Campos que dejan de tener sentido sin Internet Service — se fuerzan a "No"
# (o al sentinel, para Internet Type) cuando Internet Service = "No".
INTERNET_DEPENDENT_FIELDS = [
    "internet_type",
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
]

# Agrupación puramente visual de los 37 campos en secciones del formulario.
# Un test (test_prediction_form.py) verifica que cubre exactamente los
# nombres que build_field_specs() produce, para detectar drift si el schema
# de CustomerFeatures cambia.
FIELD_GROUPS: dict[str, list[str]] = {
    "Datos personales": [
        "gender",
        "age",
        "under_30",
        "senior_citizen",
        "married",
        "dependents",
        "number_of_dependents",
        "city",
        "population",
    ],
    "Cuenta y contrato": [
        "referred_a_friend",
        "number_of_referrals",
        "tenure_in_months",
        "offer",
        "contract",
        "paperless_billing",
        "payment_method",
        "satisfaction_score",
    ],
    "Servicios": [
        "phone_service",
        "multiple_lines",
        "internet_service",
        "internet_type",
        "avg_monthly_gb_download",
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
        "streaming_music",
        "unlimited_data",
    ],
    "Cargos": [
        "avg_monthly_long_distance_charges",
        "monthly_charge",
        "total_charges",
        "total_refunds",
        "total_extra_data_charges",
        "total_long_distance_charges",
        "total_revenue",
    ],
}

# Pisos mínimos CONDICIONALES: (campo_condición, valor_que_activa_el_piso,
# mínimo). Verificado contra data/processed/train.parquet — ninguna fila
# tiene, por ejemplo, Referred a Friend="Yes" con Number of Referrals=0
# (mínimo real 1). Cuando la condición NO se cumple, el campo ya se fuerza
# a 0 en apply_dependency_rules()/se bloquea en locked_fields().
CONDITIONAL_MINIMUMS: dict[str, tuple[str, str, float]] = {
    "number_of_referrals": ("referred_a_friend", "Yes", 1),
    "number_of_dependents": ("dependents", "Yes", 1),
    "avg_monthly_long_distance_charges": ("phone_service", "Yes", 1.0),
    "total_long_distance_charges": ("phone_service", "Yes", 1.0),
    "avg_monthly_gb_download": ("internet_service", "Yes", 1),
}

# Pisos mínimos INCONDICIONALES: en el dataset real ningún cliente activo
# tiene Monthly Charge/Total Charges/Total Revenue en 0 (todos pagan algo),
# a diferencia de Total Refunds/Total Extra Data Charges, que sí son 0 en
# ~90% de los casos y se dejan libres.
UNCONDITIONAL_MINIMUMS: dict[str, float] = {
    "monthly_charge": 18.55,
    "total_charges": 18.80,
    "total_revenue": 21.36,
}

# Rangos reales de data/processed/train.parquet (no inventados) — usados
# solo por el generador de valores aleatorios, para que el "cliente
# aleatorio" sea representativo del dataset real.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (19, 80),
    "number_of_dependents": (0, 9),
    "population": (11, 105285),
    "number_of_referrals": (0, 11),
    "tenure_in_months": (1, 72),
    "avg_monthly_long_distance_charges": (0.0, 49.99),
    "avg_monthly_gb_download": (0, 85),
    "monthly_charge": (18.55, 118.75),
    "total_charges": (18.80, 8684.80),
    "total_refunds": (0.0, 49.57),
    "total_extra_data_charges": (0, 150),
    "total_long_distance_charges": (0.0, 3564.72),
    "total_revenue": (21.36, 11979.34),
    "satisfaction_score": (1, 5),
}


def field_key(name: str) -> str:
    """Key de session_state/widget para el campo `name`."""
    return f"prediction_field_{name}"


def build_specs(city_choices: list[str]) -> list[FieldSpec]:
    """Igual que build_field_specs(), pero con `city` convertido de texto
    libre a un select acotado a las ciudades reales del dataset."""
    specs = build_field_specs()
    return [
        replace(s, kind="select", choices=tuple(city_choices))
        if s.name == "city"
        else s
        for s in specs
    ]


def default_value(spec: FieldSpec) -> object:
    if spec.kind == "select":
        return NONE_SENTINEL if spec.nullable else spec.choices[0]
    if spec.kind == "int":
        low, _ = NUMERIC_RANGES.get(spec.name, (0, 0))
        return int(low)
    if spec.kind == "float":
        low, _ = NUMERIC_RANGES.get(spec.name, (0.0, 0.0))
        return float(low)
    return spec.choices[0] if spec.choices else ""


def random_value(spec: FieldSpec, rng: random.Random) -> object:
    """Valor aleatorio válido para `spec`. No necesita conocer las reglas de
    dependencia entre campos — apply_dependency_rules() se vuelve a correr
    en cada rerun y corrige cualquier combinación inconsistente que salga
    acá (ej. Internet Service=No con Online Security=Yes)."""
    if spec.kind == "select":
        choices = (
            [NONE_SENTINEL, *spec.choices] if spec.nullable else list(spec.choices)
        )
        return rng.choice(choices)
    if spec.kind == "int":
        low, high = NUMERIC_RANGES[spec.name]
        return rng.randint(int(low), int(high))
    if spec.kind == "float":
        low, high = NUMERIC_RANGES[spec.name]
        return round(rng.uniform(low, high), 2)
    return spec.choices[0] if spec.choices else ""


def numeric_min_value(name: str, values: dict[str, object]) -> float | int | None:
    """Piso a pasarle a st.number_input(min_value=...) para `name`, dado el
    estado actual del formulario. None si `name` no tiene piso (no aparece
    en ninguno de los dos diccionarios de arriba)."""
    if name in UNCONDITIONAL_MINIMUMS:
        return UNCONDITIONAL_MINIMUMS[name]
    if name in CONDITIONAL_MINIMUMS:
        source_field, trigger_value, minimum = CONDITIONAL_MINIMUMS[name]
        return minimum if values.get(source_field) == trigger_value else 0
    return None


def _is_below_minimum(value: object, minimum: float) -> bool:
    return isinstance(value, int | float) and value < minimum


def apply_dependency_rules(values: dict[str, object]) -> dict[str, object]:
    """Corrige `values` para que sea una combinación que existe en el
    dataset real. Idempotente: correrla dos veces da el mismo resultado."""
    values = dict(values)

    if values.get("internet_service") == "No":
        values["internet_type"] = NONE_SENTINEL
        values["avg_monthly_gb_download"] = 0
        for name in INTERNET_DEPENDENT_FIELDS:
            if name != "internet_type":
                values[name] = "No"

    if values.get("phone_service") == "No":
        values["multiple_lines"] = "No"
        values["avg_monthly_long_distance_charges"] = 0.0
        values["total_long_distance_charges"] = 0.0

    if values.get("dependents") == "No":
        values["number_of_dependents"] = 0

    if values.get("referred_a_friend") == "No":
        values["number_of_referrals"] = 0

    # Pisos mínimos: si el campo quedó en 0 (valor por defecto o generado al
    # azar) pero la condición que lo habilita está activa, se sube al
    # mínimo real de train.parquet en vez de dejarlo en un estado que no
    # existe en el dataset (ver CONDITIONAL_MINIMUMS/UNCONDITIONAL_MINIMUMS).
    for name in CONDITIONAL_MINIMUMS:
        minimum = numeric_min_value(name, values)
        if minimum and _is_below_minimum(values.get(name), minimum):
            values[name] = minimum
    for name, minimum in UNCONDITIONAL_MINIMUMS.items():
        if _is_below_minimum(values.get(name), minimum):
            values[name] = minimum

    age = values.get("age")
    if isinstance(age, int | float):
        values["under_30"] = "Yes" if age < 30 else "No"

    return values


def locked_fields(values: dict[str, object]) -> set[str]:
    """Nombres de campos que deben renderizarse `disabled=True` dado el
    estado actual de `values` (después de apply_dependency_rules)."""
    locked = {"under_30"}  # siempre derivado de age, nunca editable directo

    if values.get("internet_service") == "No":
        locked.update(INTERNET_DEPENDENT_FIELDS)
        locked.add("avg_monthly_gb_download")
    if values.get("phone_service") == "No":
        locked.add("multiple_lines")
        locked.add("avg_monthly_long_distance_charges")
        locked.add("total_long_distance_charges")
    if values.get("dependents") == "No":
        locked.add("number_of_dependents")
    if values.get("referred_a_friend") == "No":
        locked.add("number_of_referrals")

    return locked


def to_payload(values: dict[str, object]) -> dict[str, object]:
    """Estado interno del formulario (con NONE_SENTINEL) -> dict que espera
    `CustomerFeatures.model_validate` (None real en vez del sentinel de UI)."""
    return {name: (None if v == NONE_SENTINEL else v) for name, v in values.items()}
