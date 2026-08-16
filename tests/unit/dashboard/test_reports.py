from pathlib import Path

from churn_mlops.dashboard.reports import read_csv_report, read_text_report


def test_read_text_report_returns_none_when_missing(tmp_path: Path) -> None:
    result = read_text_report(tmp_path / "does_not_exist.md")

    assert result is None


def test_read_text_report_returns_content_when_present(tmp_path: Path) -> None:
    report_path = tmp_path / "summary.md"
    report_path.write_text("# Título\ncontenido", encoding="utf-8")

    result = read_text_report(report_path)

    assert result == "# Título\ncontenido"


def test_read_csv_report_returns_none_when_missing(tmp_path: Path) -> None:
    result = read_csv_report(tmp_path / "does_not_exist.csv")

    assert result is None


def test_read_csv_report_returns_dataframe_when_present(tmp_path: Path) -> None:
    report_path = tmp_path / "results.csv"
    report_path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = read_csv_report(report_path)

    assert result is not None
    assert result.to_dict(orient="records") == [{"a": 1, "b": 2}]
