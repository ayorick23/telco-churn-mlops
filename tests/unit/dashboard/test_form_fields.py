from churn_mlops.dashboard.form_fields import build_field_specs


def test_build_field_specs_covers_all_37_raw_columns() -> None:
    specs = build_field_specs()

    assert len(specs) == 37


def test_build_field_specs_marks_yes_no_columns_as_select() -> None:
    specs = {s.name: s for s in build_field_specs()}

    assert specs["contract"].kind == "select"
    assert specs["contract"].choices == ("Month-to-Month", "One Year", "Two Year")
    assert specs["contract"].nullable is False


def test_build_field_specs_marks_offer_as_nullable() -> None:
    specs = {s.name: s for s in build_field_specs()}

    assert specs["offer"].nullable is True
    assert "Offer A" in specs["offer"].choices


def test_build_field_specs_infers_numeric_kinds() -> None:
    specs = {s.name: s for s in build_field_specs()}

    assert specs["age"].kind == "int"
    assert specs["monthly_charge"].kind == "float"
    assert specs["city"].kind == "text"


def test_build_field_specs_labels_use_original_column_names() -> None:
    specs = {s.name: s for s in build_field_specs()}

    assert specs["under_30"].label == "Under 30"
