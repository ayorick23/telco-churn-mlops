"""Lógica pura de decisión champion/challenger (ADR 0012) — sin MLflow, sin
red. Recibe métricas ya calculadas (agregado + por segmento) y aplica el
criterio de promoción de configs/registry.yaml: el challenger promueve solo
si (a) gana el agregado por >= promotion_margin Y (b) no retrocede en ningún
segmento más de segment_regression_tolerance. Testeable en aislamiento,
mismo rol que drift_detection.py/evaluation.py en fases previas.

No maneja el caso bootstrap (sin champion todavía) — eso lo decide
registry/run_promotion.py antes de llamar acá, porque no hay
champion_metric/segment_metric con qué comparar."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    aggregate_margin: float
    segment_regressions: dict[str, float] = field(default_factory=dict)


def compare_aggregate(
    champion_metric: float, challenger_metric: float, promotion_margin: float
) -> tuple[bool, float]:
    """Devuelve (gana, delta) donde delta = challenger - champion; gana si
    delta >= promotion_margin."""
    delta = challenger_metric - champion_metric
    return delta >= promotion_margin, delta


def find_segment_regressions(
    champion_segment_metric: dict[str, float],
    challenger_segment_metric: dict[str, float],
    segment_regression_tolerance: float,
) -> dict[str, float]:
    """Segmentos donde challenger - champion es menor que -tolerance. Solo
    compara segmentos presentes en ambos (un segmento nuevo/ausente en una de
    las dos corridas no bloquea la promoción). Dict vacío = sin regresiones
    bloqueantes."""
    regressions = {}
    for segment, champion_value in champion_segment_metric.items():
        if segment not in challenger_segment_metric:
            continue
        delta = challenger_segment_metric[segment] - champion_value
        if delta < -segment_regression_tolerance:
            regressions[segment] = delta
    return regressions


def decide_promotion(
    champion_metric: float,
    challenger_metric: float,
    champion_segment_metric: dict[str, float],
    challenger_segment_metric: dict[str, float],
    registry_config: dict[str, Any],
) -> PromotionDecision:
    """Combina compare_aggregate + find_segment_regressions: promueve solo si
    el challenger gana el agregado por el margen configurado Y no retrocede
    en ningún segmento más allá de la tolerancia configurada."""
    promotion_margin = registry_config["promotion_margin"]
    segment_regression_tolerance = registry_config["segment_regression_tolerance"]

    wins_aggregate, aggregate_margin = compare_aggregate(
        champion_metric, challenger_metric, promotion_margin
    )
    segment_regressions = find_segment_regressions(
        champion_segment_metric, challenger_segment_metric, segment_regression_tolerance
    )

    if not wins_aggregate:
        return PromotionDecision(
            promote=False,
            reason=(
                f"challenger no alcanza promotion_margin={promotion_margin} "
                f"(delta agregado={aggregate_margin:.4f})"
            ),
            aggregate_margin=aggregate_margin,
        )

    if segment_regressions:
        return PromotionDecision(
            promote=False,
            reason=(
                "challenger gana el agregado pero retrocede en segmento(s) "
                f"más allá de segment_regression_tolerance="
                f"{segment_regression_tolerance}: {list(segment_regressions)}"
            ),
            aggregate_margin=aggregate_margin,
            segment_regressions=segment_regressions,
        )

    return PromotionDecision(
        promote=True,
        reason=(
            f"challenger gana el agregado por {aggregate_margin:.4f} "
            "sin regresiones de segmento bloqueantes"
        ),
        aggregate_margin=aggregate_margin,
    )
