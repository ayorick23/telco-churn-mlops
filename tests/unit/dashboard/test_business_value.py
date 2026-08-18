from churn_mlops.dashboard import business_value as bv

TERMS = {"Month-to-Month": 1, "One Year": 12, "Two Year": 24}


def test_remaining_contract_months_for_month_to_month_is_always_the_term() -> None:
    assert bv.remaining_contract_months("Month-to-Month", 0, TERMS) == 1
    assert bv.remaining_contract_months("Month-to-Month", 47, TERMS) == 1


def test_remaining_contract_months_counts_down_within_the_current_cycle() -> None:
    assert bv.remaining_contract_months("One Year", 3, TERMS) == 9
    assert bv.remaining_contract_months("Two Year", 5, TERMS) == 19


def test_remaining_contract_months_is_a_full_term_right_after_a_renewal() -> None:
    assert bv.remaining_contract_months("One Year", 12, TERMS) == 12
    assert bv.remaining_contract_months("Two Year", 24, TERMS) == 24
    assert bv.remaining_contract_months("One Year", 0, TERMS) == 12


def test_estimate_retention_value_computes_expected_loss_and_net_benefit() -> None:
    result = bv.estimate_retention_value(
        contract="One Year",
        tenure_in_months=3,
        monthly_charge=100.0,
        churn_probability=0.5,
        contract_term_months=TERMS,
        retention_campaign_cost=50.0,
    )

    assert result.remaining_months == 9
    assert result.value_at_risk == 900.0
    assert result.expected_loss == 450.0
    assert result.net_benefit == 400.0
    assert result.worth_retaining is True


def test_estimate_retention_value_flags_not_worth_retaining_when_cost_exceeds_loss() -> (
    None
):
    result = bv.estimate_retention_value(
        contract="Month-to-Month",
        tenure_in_months=10,
        monthly_charge=20.0,
        churn_probability=0.1,
        contract_term_months=TERMS,
        retention_campaign_cost=50.0,
    )

    assert result.expected_loss == 2.0
    assert result.net_benefit == -48.0
    assert result.worth_retaining is False
