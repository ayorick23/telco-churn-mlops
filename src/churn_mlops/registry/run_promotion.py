"""Orquesta Fase 6 (paso 2 de 2): compara el challenger (última versión
registrada de churn-lightgbm) contra el champion actual (alias `champion`)
sobre test.parquet completo + cada segmento de configs/registry.yaml, decide
si promueve (ADR 0012) y, si corresponde, mueve el alias.

Los bytes de los pipelines NO se descargan del Model Registry (ADR 0012: el
proxy de artifacts de MLflow en DagsHub no es confiable para descargas). El
challenger se lee del `.pkl` que run_training.py/run_retrain.py acaban de
dejar en disco (`configs/training.yaml:model_output_path`); el champion se
lee de una copia estable versionada por DVC
(`configs/registry.yaml:champion_pipeline_path`), actualizada acá mismo cada
vez que se promueve. El Registry + el alias `champion` siguen siendo la
fuente de verdad de METADATA (qué versión es cuál) — solo dejan de ser el
mecanismo para traer los bytes del modelo.

Caso bootstrap: si no existe ningún champion todavía, promueve el challenger
sin comparar — no hay contra qué comparar. Loguea un run de resumen a MLflow
con la decisión completa y guarda un reporte CSV+MD en reports/registry/
(mismo patrón que save_comparison_report/save_summary_report de Fase 3/5).
No es stage de dvc.yaml (mismo precedente, ADR 0009/0011). Se corre a mano,
después de `dvc repro train_model` o de `training.run_retrain`:

    uv run python -m churn_mlops.registry.run_promotion

Nota operativa: si este script promueve, `champion_pipeline_path` cambia por
fuera de `dvc.yaml` — hace falta `dvc add` + `dvc push` después para dejarlo
versionado y disponible para otra máquina/colaborador (ver ADR 0012, mismo
tipo de fricción que `dvc commit train_model` tras `run_retrain.py`).
"""

import io
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.entities.model_registry import ModelVersion
from sklearn.pipeline import Pipeline

# La consola de Windows usa cp1252 por defecto, que no soporta los emojis que
# mlflow imprime al terminar un run (ej. 🏃). Sin esto, el script crashea después
# de loguear el run exitosamente. El isinstance narrowa sys.stdout a
# TextIOWrapper para mypy (TextIOBase no declara reconfigure).
if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from churn_mlops.config import load_yaml_config
from churn_mlops.registry.comparison import PromotionDecision, decide_promotion
from churn_mlops.registry.mlflow_client import (
    get_champion_version,
    get_latest_version,
    promote_challenger,
)
from churn_mlops.registry.segment_evaluation import compute_segment_metrics
from churn_mlops.training.dataset import load_processed_datasets
from churn_mlops.training.evaluation import compute_classification_metrics
from churn_mlops.training.run_training import save_pipeline_artifact


def configure_mlflow(registry_config: dict[str, Any]) -> None:
    """Carga .env y apunta MLflow al server de DagsHub — idéntico patrón a
    las otras fases."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(registry_config["mlflow"]["experiment_name"])


def load_local_pipeline(path: str | Path) -> Pipeline:
    """Carga un Pipeline serializado por joblib desde disco (ADR 0012) —
    usado tanto para el challenger recién entrenado como para la copia
    estable del champion. Es la contraparte de lectura de
    training/run_training.py::save_pipeline_artifact."""
    pipeline: Pipeline = joblib.load(Path(path))
    return pipeline


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    segment_columns: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Un solo predict/predict_proba, agregado (compute_classification_metrics)
    más por segmento (segment_evaluation.compute_segment_metrics) — evita
    duplicar el forward pass entre champion y challenger."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    aggregate_metrics = compute_classification_metrics(
        y_test.to_numpy(), y_pred, y_proba
    )
    segment_metrics = compute_segment_metrics(
        X_test, y_test.to_numpy(), y_pred, y_proba, segment_columns
    )
    return aggregate_metrics, segment_metrics


def save_promotion_report(
    challenger_version: ModelVersion,
    champion_version: ModelVersion | None,
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    decision: PromotionDecision | None,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Guarda la comparación agregada (CSV) y el veredicto (Markdown) en
    reports/registry/."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "role": "challenger",
            "version": challenger_version.version,
            **challenger_metrics,
        }
    ]
    if champion_version is not None and champion_metrics is not None:
        rows.append(
            {
                "role": "champion",
                "version": champion_version.version,
                **champion_metrics,
            }
        )
    table = pd.DataFrame(rows)

    csv_path = output_path / "promotion_results.csv"
    table.to_csv(csv_path, index=False)

    if decision is None:
        verdict = (
            f"Bootstrap: no había champion todavía — se promovió la versión "
            f"{challenger_version.version} de churn-lightgbm sin comparar."
        )
    else:
        verdict = f"Promovido: **{decision.promote}**. {decision.reason}"

    summary_path = output_path / "promotion_summary.md"
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    table_rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    summary_path.write_text(
        "# Fase 6 — Comparación champion/challenger\n\n"
        f"{verdict}\n\n"
        f"{header}\n{separator}\n" + "\n".join(table_rows) + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path


def main() -> None:
    registry_config = load_yaml_config("configs/registry.yaml")
    data_config = load_yaml_config("configs/data.yaml")
    training_config = load_yaml_config("configs/training.yaml")
    configure_mlflow(registry_config)

    model_name = registry_config["registered_model_name"]
    champion_alias = registry_config["champion_alias"]
    segment_columns = registry_config["segment_columns"]
    champion_pipeline_path = Path(registry_config["champion_pipeline_path"])

    client = mlflow.MlflowClient()
    challenger_version = get_latest_version(client, model_name)
    champion_version = get_champion_version(client, model_name, champion_alias)

    if champion_version is not None and not champion_pipeline_path.exists():
        raise FileNotFoundError(
            f"Existe un champion registrado (alias '{champion_alias}') en el "
            f"Model Registry pero no {champion_pipeline_path} en disco — "
            "correr `dvc pull` primero."
        )

    _, X_test, _, y_test = load_processed_datasets(data_config)

    challenger_pipeline = load_local_pipeline(training_config["model_output_path"])
    challenger_metrics, challenger_segments = evaluate_pipeline(
        challenger_pipeline, X_test, y_test, segment_columns
    )

    with mlflow.start_run(run_name="promotion-decision"):
        mlflow.log_param("challenger_version", challenger_version.version)
        mlflow.log_metrics(
            {f"challenger_{k}": v for k, v in challenger_metrics.items()}
        )
        mlflow.log_dict(challenger_segments, "challenger_segment_metrics.json")

        if champion_version is None:
            mlflow.log_param("bootstrap", True)
            promote_challenger(
                client, model_name, champion_alias, challenger_version.version
            )
            save_pipeline_artifact(challenger_pipeline, champion_pipeline_path)
            csv_path, summary_path = save_promotion_report(
                challenger_version,
                None,
                challenger_metrics,
                None,
                None,
                registry_config["reports_dir"],
            )
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(summary_path))
            print(
                f"Bootstrap: se promovió la versión {challenger_version.version} "
                f"de {model_name} como {champion_alias} (no había champion previo)."
            )
            return

        champion_pipeline = load_local_pipeline(champion_pipeline_path)
        champion_metrics, champion_segments = evaluate_pipeline(
            champion_pipeline, X_test, y_test, segment_columns
        )
        mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
        mlflow.log_dict(champion_segments, "champion_segment_metrics.json")

        promotion_metric = registry_config["promotion_metric"]
        decision = decide_promotion(
            champion_metrics[promotion_metric],
            challenger_metrics[promotion_metric],
            {k: v[promotion_metric] for k, v in champion_segments.items()},
            {k: v[promotion_metric] for k, v in challenger_segments.items()},
            registry_config,
        )
        mlflow.log_param("promote", decision.promote)
        mlflow.log_param("reason", decision.reason)
        mlflow.log_metric("aggregate_margin", decision.aggregate_margin)

        if decision.promote:
            promote_challenger(
                client, model_name, champion_alias, challenger_version.version
            )
            save_pipeline_artifact(challenger_pipeline, champion_pipeline_path)

        csv_path, summary_path = save_promotion_report(
            challenger_version,
            champion_version,
            challenger_metrics,
            champion_metrics,
            decision,
            registry_config["reports_dir"],
        )
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(summary_path))

        print(
            f"Challenger versión {challenger_version.version} vs. champion versión {champion_version.version}"
        )
        print(f"Decisión: promote={decision.promote} — {decision.reason}")


if __name__ == "__main__":
    main()
