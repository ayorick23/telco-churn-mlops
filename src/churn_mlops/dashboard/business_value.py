"""Estimación de valor de negocio del churn — KPI mostrado en la página de
Predicción del dashboard (decisión de alcance: solo dashboard, no toca
training/registry). Funciones puras, sin dependencia de `streamlit`, igual
que `prediction_form.py`.

Supuestos de negocio (no derivados del dataset, configurables en
`configs/dashboard.yaml::business_value`):
- El valor en riesgo de un cliente es su Monthly Charge multiplicado por los
  meses que quedan del ciclo de contrato actual. Month-to-Month no tiene
  plazo fijo, así que se trata como el término configurado (por defecto 1
  mes, el aviso mínimo de baja). One Year/Two Year se renuevan cada `term`
  meses — lo que falta se deriva de `tenure_in_months % term`, no de un
  plazo fijo desde el alta.
- La pérdida esperada es `churn_probability × valor_en_riesgo`.
- Conviene retener si la pérdida esperada supera el costo de la campaña de
  retención (beneficio neto > 0).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionValue:
    remaining_months: int
    value_at_risk: float
    expected_loss: float
    net_benefit: float
    worth_retaining: bool


def remaining_contract_months(
    contract: str, tenure_in_months: int, contract_term_months: dict[str, int]
) -> int:
    term = contract_term_months[contract]
    if term <= 1:
        return term
    remainder = tenure_in_months % term
    return term - remainder if remainder else term


def estimate_retention_value(
    *,
    contract: str,
    tenure_in_months: int,
    monthly_charge: float,
    churn_probability: float,
    contract_term_months: dict[str, int],
    retention_campaign_cost: float,
) -> RetentionValue:
    remaining = remaining_contract_months(
        contract, tenure_in_months, contract_term_months
    )
    value_at_risk = monthly_charge * remaining
    expected_loss = churn_probability * value_at_risk
    net_benefit = expected_loss - retention_campaign_cost
    return RetentionValue(
        remaining_months=remaining,
        value_at_risk=value_at_risk,
        expected_loss=expected_loss,
        net_benefit=net_benefit,
        worth_retaining=net_benefit > 0,
    )
