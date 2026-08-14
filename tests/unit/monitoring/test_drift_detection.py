import numpy as np
import pandas as pd
from churn_mlops.monitoring.drift_detection import (
    run_drift_report,
    save_drift_report,
    summarize_drift_result,
)


def _make_reference(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "numeric_a": rng.normal(0, 1, n),
            "category_b": rng.choice(["x", "y"], n),
        }
    )


def test_summarize_drift_result_low_share_when_no_drift() -> None:
    reference_df = _make_reference()
    current_df = _make_reference()

    result = run_drift_report(reference_df, current_df, drift_share=0.5)
    summary = summarize_drift_result(result)

    assert summary["dataset_drift_share"] < 0.5
    assert summary["n_drifted_columns"] == 0.0


def test_summarize_drift_result_high_share_when_obvious_drift() -> None:
    reference_df = _make_reference()
    current_df = reference_df.copy()
    current_df["numeric_a"] = (
        current_df["numeric_a"] + 10 * reference_df["numeric_a"].std()
    )

    result = run_drift_report(reference_df, current_df, drift_share=0.5)
    summary = summarize_drift_result(result)

    assert summary["n_drifted_columns"] >= 1.0
    assert summary["dataset_drift_share"] >= 0.5


def test_save_drift_report_writes_non_empty_html_and_json(tmp_path) -> None:
    reference_df = _make_reference()
    result = run_drift_report(reference_df, reference_df, drift_share=0.5)

    html_path, json_path = save_drift_report(result, tmp_path, "test_batch")

    assert html_path.exists() and html_path.stat().st_size > 0
    assert json_path.exists() and json_path.stat().st_size > 0
