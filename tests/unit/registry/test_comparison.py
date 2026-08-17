import pytest

from churn_mlops.registry.comparison import (
    compare_aggregate,
    decide_promotion,
    find_segment_regressions,
)

REGISTRY_CONFIG = {"promotion_margin": 0.01, "segment_regression_tolerance": 0.03}


def test_compare_aggregate_wins_when_delta_meets_margin() -> None:
    wins, delta = compare_aggregate(0.80, 0.82, promotion_margin=0.01)

    assert wins is True
    assert delta == pytest.approx(0.02)


def test_compare_aggregate_loses_when_delta_below_margin() -> None:
    wins, delta = compare_aggregate(0.80, 0.805, promotion_margin=0.01)

    assert wins is False


def test_compare_aggregate_boundary_is_inclusive() -> None:
    wins, _ = compare_aggregate(0.80, 0.81, promotion_margin=0.01)

    assert wins is True


def test_find_segment_regressions_flags_only_violations_beyond_tolerance() -> None:
    champion = {"Contract=Month-to-Month": 0.90, "Contract=One Year": 0.70}
    challenger = {"Contract=Month-to-Month": 0.87, "Contract=One Year": 0.71}

    regressions = find_segment_regressions(
        champion, challenger, segment_regression_tolerance=0.03
    )

    assert list(regressions.keys()) == ["Contract=Month-to-Month"]


def test_find_segment_regressions_within_tolerance_is_not_a_regression() -> None:
    champion = {"Contract=Month-to-Month": 0.90}
    challenger = {"Contract=Month-to-Month": 0.88}  # delta = -0.02, tolerance = 0.03

    regressions = find_segment_regressions(
        champion, challenger, segment_regression_tolerance=0.03
    )

    assert regressions == {}


def test_decide_promotion_true_when_wins_aggregate_and_no_regressions() -> None:
    decision = decide_promotion(
        champion_metric=0.80,
        challenger_metric=0.83,
        champion_segment_metric={"Contract=Month-to-Month": 0.90},
        challenger_segment_metric={"Contract=Month-to-Month": 0.90},
        registry_config=REGISTRY_CONFIG,
    )

    assert decision.promote is True


def test_decide_promotion_false_when_margin_not_met() -> None:
    decision = decide_promotion(
        champion_metric=0.80,
        challenger_metric=0.805,
        champion_segment_metric={},
        challenger_segment_metric={},
        registry_config=REGISTRY_CONFIG,
    )

    assert decision.promote is False
    assert "promotion_margin" in decision.reason


def test_decide_promotion_false_when_aggregate_wins_but_segment_regresses() -> None:
    decision = decide_promotion(
        champion_metric=0.80,
        challenger_metric=0.85,
        champion_segment_metric={"Contract=Two Year": 0.95},
        challenger_segment_metric={"Contract=Two Year": 0.80},
        registry_config=REGISTRY_CONFIG,
    )

    assert decision.promote is False
    assert decision.segment_regressions.keys() == {"Contract=Two Year"}
    assert decision.segment_regressions["Contract=Two Year"] == pytest.approx(-0.15)


def test_decide_promotion_true_when_segment_regresses_within_tolerance() -> None:
    decision = decide_promotion(
        champion_metric=0.80,
        challenger_metric=0.85,
        champion_segment_metric={"Contract=Two Year": 0.95},
        challenger_segment_metric={"Contract=Two Year": 0.93},  # delta = -0.02
        registry_config=REGISTRY_CONFIG,
    )

    assert decision.promote is True
