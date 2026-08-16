import random

from churn_mlops.dashboard import prediction_form as pf
from churn_mlops.dashboard.form_fields import build_field_specs


def test_field_groups_cover_exactly_the_37_spec_names() -> None:
    spec_names = {s.name for s in build_field_specs()}
    grouped_names = {name for names in pf.FIELD_GROUPS.values() for name in names}

    assert grouped_names == spec_names


def test_build_specs_converts_city_to_bounded_select() -> None:
    specs = {s.name: s for s in pf.build_specs(["Acton", "Bell"])}

    assert specs["city"].kind == "select"
    assert specs["city"].choices == ("Acton", "Bell")


def test_default_value_for_nullable_select_is_the_none_sentinel() -> None:
    specs = {s.name: s for s in build_field_specs()}

    assert pf.default_value(specs["offer"]) == pf.NONE_SENTINEL


def test_default_value_for_non_nullable_select_is_first_choice() -> None:
    specs = {s.name: s for s in build_field_specs()}

    assert pf.default_value(specs["contract"]) == "Month-to-Month"


def test_random_value_stays_within_curated_numeric_range() -> None:
    specs = {s.name: s for s in build_field_specs()}
    rng = random.Random(0)

    for _ in range(20):
        value = pf.random_value(specs["satisfaction_score"], rng)
        assert 1 <= value <= 5


def test_random_value_for_select_is_one_of_the_choices() -> None:
    specs = {s.name: s for s in build_field_specs()}
    rng = random.Random(0)

    value = pf.random_value(specs["contract"], rng)

    assert value in specs["contract"].choices


def test_apply_dependency_rules_locks_internet_addons_when_no_internet() -> None:
    values = {"internet_service": "No", "online_security": "Yes", "streaming_tv": "Yes"}

    result = pf.apply_dependency_rules(values)

    assert result["internet_type"] == pf.NONE_SENTINEL
    assert result["online_security"] == "No"
    assert result["streaming_tv"] == "No"


def test_apply_dependency_rules_zeroes_dependents_count() -> None:
    values = {"dependents": "No", "number_of_dependents": 4}

    result = pf.apply_dependency_rules(values)

    assert result["number_of_dependents"] == 0


def test_apply_dependency_rules_bumps_referrals_above_zero_when_referred() -> None:
    values = {"referred_a_friend": "Yes", "number_of_referrals": 0}

    result = pf.apply_dependency_rules(values)

    assert result["number_of_referrals"] == 1


def test_apply_dependency_rules_bumps_dependents_above_zero_when_dependents_yes() -> (
    None
):
    values = {"dependents": "Yes", "number_of_dependents": 0}

    result = pf.apply_dependency_rules(values)

    assert result["number_of_dependents"] == 1


def test_apply_dependency_rules_zeroes_long_distance_charges_without_phone() -> None:
    values = {
        "phone_service": "No",
        "avg_monthly_long_distance_charges": 12.5,
        "total_long_distance_charges": 800.0,
    }

    result = pf.apply_dependency_rules(values)

    assert result["avg_monthly_long_distance_charges"] == 0.0
    assert result["total_long_distance_charges"] == 0.0


def test_apply_dependency_rules_bumps_long_distance_charges_above_zero_with_phone() -> (
    None
):
    values = {
        "phone_service": "Yes",
        "avg_monthly_long_distance_charges": 0.0,
        "total_long_distance_charges": 0.0,
    }

    result = pf.apply_dependency_rules(values)

    assert result["avg_monthly_long_distance_charges"] > 0
    assert result["total_long_distance_charges"] > 0


def test_apply_dependency_rules_zeroes_gb_download_without_internet() -> None:
    values = {"internet_service": "No", "avg_monthly_gb_download": 40}

    result = pf.apply_dependency_rules(values)

    assert result["avg_monthly_gb_download"] == 0


def test_apply_dependency_rules_bumps_gb_download_above_zero_with_internet() -> None:
    values = {"internet_service": "Yes", "avg_monthly_gb_download": 0}

    result = pf.apply_dependency_rules(values)

    assert result["avg_monthly_gb_download"] > 0


def test_apply_dependency_rules_enforces_unconditional_charge_floors() -> None:
    values = {"monthly_charge": 0.0, "total_charges": 0.0, "total_revenue": 0.0}

    result = pf.apply_dependency_rules(values)

    assert result["monthly_charge"] == pf.UNCONDITIONAL_MINIMUMS["monthly_charge"]
    assert result["total_charges"] == pf.UNCONDITIONAL_MINIMUMS["total_charges"]
    assert result["total_revenue"] == pf.UNCONDITIONAL_MINIMUMS["total_revenue"]


def test_apply_dependency_rules_never_forces_refunds_or_extra_data_above_zero() -> None:
    values = {"total_refunds": 0.0, "total_extra_data_charges": 0}

    result = pf.apply_dependency_rules(values)

    assert result["total_refunds"] == 0.0
    assert result["total_extra_data_charges"] == 0


def test_apply_dependency_rules_derives_under_30_from_age() -> None:
    assert pf.apply_dependency_rules({"age": 25})["under_30"] == "Yes"
    assert pf.apply_dependency_rules({"age": 45})["under_30"] == "No"


def test_apply_dependency_rules_is_idempotent() -> None:
    values = {"internet_service": "No", "age": 20, "dependents": "No"}

    once = pf.apply_dependency_rules(values)
    twice = pf.apply_dependency_rules(once)

    assert once == twice


def test_locked_fields_always_locks_under_30() -> None:
    assert "under_30" in pf.locked_fields({})


def test_locked_fields_locks_internet_addons_only_when_no_internet() -> None:
    locked_without_internet = pf.locked_fields({"internet_service": "No"})
    locked_with_internet = pf.locked_fields({"internet_service": "Yes"})

    assert "online_security" in locked_without_internet
    assert "online_security" not in locked_with_internet
    assert "avg_monthly_gb_download" in locked_without_internet
    assert "avg_monthly_gb_download" not in locked_with_internet


def test_locked_fields_locks_long_distance_charges_only_without_phone() -> None:
    locked_without_phone = pf.locked_fields({"phone_service": "No"})
    locked_with_phone = pf.locked_fields({"phone_service": "Yes"})

    assert "avg_monthly_long_distance_charges" in locked_without_phone
    assert "total_long_distance_charges" in locked_without_phone
    assert "avg_monthly_long_distance_charges" not in locked_with_phone


def test_numeric_min_value_is_conditional_on_source_field() -> None:
    assert (
        pf.numeric_min_value("number_of_referrals", {"referred_a_friend": "Yes"}) == 1
    )
    assert pf.numeric_min_value("number_of_referrals", {"referred_a_friend": "No"}) == 0


def test_numeric_min_value_is_unconditional_for_fixed_floor_fields() -> None:
    assert pf.numeric_min_value("monthly_charge", {}) == 18.55


def test_numeric_min_value_is_none_for_unrestricted_fields() -> None:
    assert pf.numeric_min_value("total_refunds", {}) is None


def test_to_payload_converts_none_sentinel_to_real_none() -> None:
    values = {"offer": pf.NONE_SENTINEL, "contract": "Two Year"}

    result = pf.to_payload(values)

    assert result["offer"] is None
    assert result["contract"] == "Two Year"
