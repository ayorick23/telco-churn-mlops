from pathlib import Path

import pytest

from churn_mlops.monitoring.run_retrain_check import load_drift_summary


def test_load_drift_summary_reads_real_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "drift_monitoring_results.csv"
    csv_path.write_text(
        "batch_name,intensity,dataset_drift_share\nbatch_a,0.0,0.026\n",
        encoding="utf-8",
    )

    result = load_drift_summary(csv_path)

    assert list(result["batch_name"]) == ["batch_a"]


def test_load_drift_summary_raises_with_explicit_message_when_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError, match="run_monitoring"):
        load_drift_summary(missing_path)
