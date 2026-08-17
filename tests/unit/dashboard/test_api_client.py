from typing import Any

import pytest

from churn_mlops.dashboard import api_client


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def test_predict_posts_a_list_and_returns_first_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: Any, timeout: int) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse([{"churn_probability": 0.5}])

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.predict("http://localhost:8000", {"Age": 30})

    assert captured["url"] == "http://localhost:8000/predict"
    assert captured["json"] == [{"Age": 30}]
    assert result == {"churn_probability": 0.5}


def test_explain_posts_a_single_object(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: Any, timeout: int) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"shap_values": {}})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.explain("http://localhost:8000", {"Age": 30})

    assert captured["url"] == "http://localhost:8000/explain"
    assert captured["json"] == {"Age": 30}
    assert result == {"shap_values": {}}


def test_model_info_gets_the_model_info_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, timeout: int) -> _FakeResponse:
        captured["url"] = url
        return _FakeResponse({"champion_version": "3"})

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    result = api_client.model_info("http://localhost:8000")

    assert captured["url"] == "http://localhost:8000/model-info"
    assert result == {"champion_version": "3"}
