import numpy as np
import pandas as pd
import pytest
from churn_mlops.data.drift_simulation import (
    apply_categorical_reweight,
    apply_drift_specs,
    apply_numeric_shift,
    generate_drifted_batch,
    sample_reference_rows,
)


def test_sample_reference_rows_returns_requested_size_with_replacement() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})

    sampled = sample_reference_rows(df, batch_size=10, random_state=42)

    assert len(sampled) == 10
    assert list(sampled.index) == list(range(10))


def test_apply_numeric_shift_at_intensity_zero_is_identity() -> None:
    series = pd.Series([10.0, 20.0, 30.0])

    result = apply_numeric_shift(series, mean_shift=100.0, std_shift=5.0, intensity=0.0)

    pd.testing.assert_series_equal(result, series)


def test_apply_numeric_shift_at_intensity_one_matches_configured_mean_shift() -> None:
    series = pd.Series([10.0, 20.0, 30.0])

    result = apply_numeric_shift(series, mean_shift=15.0, std_shift=0.0, intensity=1.0)

    assert result.mean() == pytest.approx(series.mean() + 15.0)


def test_apply_numeric_shift_respects_clip_min() -> None:
    series = pd.Series([1.0, 2.0, 3.0])

    result = apply_numeric_shift(
        series, mean_shift=-100.0, std_shift=0.0, intensity=1.0, clip_min=0.0
    )

    assert (result >= 0.0).all()


def test_apply_categorical_reweight_at_intensity_zero_preserves_empirical_distribution() -> (
    None
):
    rng = np.random.default_rng(0)
    series = pd.Series(rng.choice(["x", "y"], size=5000, p=[0.8, 0.2]))

    result = apply_categorical_reweight(
        series, target_distribution={"x": 0.0, "y": 1.0}, intensity=0.0, random_state=1
    )

    proportions = result.value_counts(normalize=True)
    assert proportions["x"] == pytest.approx(0.8, abs=0.03)


def test_apply_categorical_reweight_at_intensity_one_matches_target_distribution() -> (
    None
):
    rng = np.random.default_rng(0)
    series = pd.Series(rng.choice(["x", "y"], size=5000, p=[0.8, 0.2]))

    result = apply_categorical_reweight(
        series, target_distribution={"x": 0.3, "y": 0.7}, intensity=1.0, random_state=1
    )

    proportions = result.value_counts(normalize=True)
    assert proportions["x"] == pytest.approx(0.3, abs=0.03)


def test_apply_drift_specs_dispatches_by_type_and_leaves_other_columns_untouched() -> (
    None
):
    df = pd.DataFrame(
        {
            "Monthly Charge": [50.0, 60.0, 70.0],
            "Contract": ["Month-to-Month", "Two Year", "One Year"],
            "Untouched": [1, 2, 3],
        }
    )
    drift_specs = {
        "Monthly Charge": {
            "type": "numeric_shift",
            "mean_shift": 10.0,
            "std_shift": 0.0,
        },
        "Contract": {
            "type": "categorical_reweight",
            "target_distribution": {"Month-to-Month": 1.0},
        },
    }

    result = apply_drift_specs(
        df, drift_specs, intensity=1.0, random_state=42, target_column="Churn Label"
    )

    assert result["Monthly Charge"].mean() == pytest.approx(
        df["Monthly Charge"].mean() + 10.0
    )
    assert (result["Contract"] == "Month-to-Month").all()
    assert result["Untouched"].tolist() == [1, 2, 3]


def test_apply_drift_specs_rejects_target_column() -> None:
    df = pd.DataFrame({"Churn Label": [0, 1]})
    drift_specs = {
        "Churn Label": {"type": "numeric_shift", "mean_shift": 1.0, "std_shift": 0.0}
    }

    with pytest.raises(ValueError):
        apply_drift_specs(
            df, drift_specs, intensity=1.0, random_state=42, target_column="Churn Label"
        )


def test_apply_drift_specs_rejects_unknown_type() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    drift_specs = {"a": {"type": "bogus"}}

    with pytest.raises(ValueError):
        apply_drift_specs(
            df, drift_specs, intensity=1.0, random_state=42, target_column="Churn Label"
        )


def test_generate_drifted_batch_is_deterministic_given_same_random_state() -> None:
    reference_df = pd.DataFrame(
        {
            "Monthly Charge": np.arange(100, dtype=float),
            "Contract": ["Month-to-Month", "Two Year"] * 50,
            "Churn Label": [0, 1] * 50,
        }
    )
    drift_config = {
        "batch_size": 20,
        "random_state": 42,
        "drift_specs": {
            "Monthly Charge": {
                "type": "numeric_shift",
                "mean_shift": 10.0,
                "std_shift": 0.0,
            },
            "Contract": {
                "type": "categorical_reweight",
                "target_distribution": {"Month-to-Month": 0.9, "Two Year": 0.1},
            },
        },
    }

    first = generate_drifted_batch(
        reference_df, drift_config, intensity=0.5, target_column="Churn Label"
    )
    second = generate_drifted_batch(
        reference_df, drift_config, intensity=0.5, target_column="Churn Label"
    )

    pd.testing.assert_frame_equal(first, second)

    different_seed_config = {**drift_config, "random_state": 43}
    third = generate_drifted_batch(
        reference_df, different_seed_config, intensity=0.5, target_column="Churn Label"
    )
    assert not first.equals(third)
