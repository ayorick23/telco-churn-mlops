"""Orquesta Fase 6 (paso 1 de 2): lee configs/retraining.yaml + el resumen
cross-batch de Fase 5 (reports/monitoring/drift_monitoring_results.csv) y
recomienda o no reentrenar (ADR 0012). NO reentrena — solo informa. Si
recomienda reentrenar, imprime los próximos comandos manuales:

    uv run python -m churn_mlops.training.run_retrain
    uv run python -m churn_mlops.registry.run_promotion

No es stage de dvc.yaml (mismo precedente que run_monitoring.py/
run_model_selection.py, ADR 0009/0011). Entrypoint:

    uv run python -m churn_mlops.monitoring.run_retrain_check
"""

import io
import sys
from pathlib import Path

import pandas as pd

# La consola de Windows usa cp1252 por defecto, que no soporta caracteres
# acentuados en el print() de más abajo ("Próximos", etc.). El isinstance
# narrowa sys.stdout a TextIOWrapper para mypy (TextIOBase no declara
# reconfigure).
if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from churn_mlops.config import load_yaml_config
from churn_mlops.monitoring.retrain_trigger import evaluate_retrain_trigger


def load_drift_summary(path: str | Path) -> pd.DataFrame:
    """Lee el CSV de resumen de Fase 5. Lanza FileNotFoundError explícito si
    todavía no corrió `run_monitoring.py`."""
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No existe {summary_path} — correr primero "
            "`uv run python -m churn_mlops.monitoring.run_monitoring` "
            "para generar el resumen de drift de Fase 5."
        )
    return pd.read_csv(summary_path)


def main() -> None:
    retraining_config = load_yaml_config("configs/retraining.yaml")
    drift_summary = load_drift_summary(retraining_config["drift_summary_path"])

    decision = evaluate_retrain_trigger(drift_summary, retraining_config)

    print(
        f"{decision.trigger_metric}={decision.observed_value:.4f} "
        f"(batch: {decision.triggering_batch}) vs. umbral={decision.threshold}"
    )
    if decision.should_retrain:
        print("Retrain recomendado: SI")
        print("Próximos pasos:")
        print("  uv run python -m churn_mlops.training.run_retrain")
        print("  uv run python -m churn_mlops.registry.run_promotion")
    else:
        print("Retrain recomendado: NO")


if __name__ == "__main__":
    main()
