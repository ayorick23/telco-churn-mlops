"""Lee los reportes estáticos que las Fases 3/5/6 ya versionan en git
(reports/model_selection_summary.md, reports/monitoring/*, reports/registry/*)
— decisión de Fase 7: el dashboard los lee directo, no hay endpoints nuevos en
la API solo para esto (ver docs/decisions/0013-serving-y-dashboard-fase-7.md).

Funciones puras de I/O de solo lectura; devuelven None si el reporte todavía
no se generó (ej. nadie corrió run_monitoring.py todavía) en vez de lanzar,
para que la UI pueda mostrar un mensaje en vez de crashear."""

from pathlib import Path

import pandas as pd


def read_text_report(path: str | Path) -> str | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    return report_path.read_text(encoding="utf-8")


def read_csv_report(path: str | Path) -> pd.DataFrame | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    return pd.read_csv(report_path)
