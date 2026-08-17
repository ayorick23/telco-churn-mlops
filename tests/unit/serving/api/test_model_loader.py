from pathlib import Path

import pytest

from churn_mlops.serving.api import model_loader


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """get_pipeline/get_serving_config son lru_cache(maxsize=1) — sin esto,
    el resultado de un test contamina el siguiente."""
    model_loader.get_pipeline.cache_clear()
    model_loader.get_serving_config.cache_clear()
    yield
    model_loader.get_pipeline.cache_clear()
    model_loader.get_serving_config.cache_clear()


def test_get_pipeline_raises_clear_error_when_champion_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_path = tmp_path / "champion_pipeline.pkl"
    monkeypatch.setattr(
        model_loader,
        "get_serving_config",
        lambda: {"model_path": str(missing_path)},
    )

    with pytest.raises(FileNotFoundError, match="dvc pull"):
        model_loader.get_pipeline()
