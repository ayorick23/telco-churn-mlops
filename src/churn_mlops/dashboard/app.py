"""Dashboard de churn (Fase 7). Levanta con:

    uv run streamlit run src/churn_mlops/dashboard/app.py

Navegación por sidebar (`st.navigation`/`st.Page`, con espacio para logo vía
`st.logo`) en vez de tabs. Cuatro páginas: Predicción (formulario reactivo
contra la API — ver `prediction_form.py` para las reglas de dependencia
entre campos), Producción y Drift (leen `reports/` directo, solo
`st.dataframe`, sin duplicar tablas markdown/dataframe — ADR 0013), y
Operaciones (botones que corren los scripts manuales de Fase 5/6 vía
subprocess, sin reimplementar su lógica)."""

import os
import random
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from churn_mlops.config import load_yaml_config
from churn_mlops.dashboard import api_client
from churn_mlops.dashboard import business_value as bv
from churn_mlops.dashboard import prediction_form as pf
from churn_mlops.dashboard.data_sources import load_known_cities
from churn_mlops.dashboard.form_fields import FieldSpec
from churn_mlops.dashboard.reports import read_csv_report, read_text_report
from churn_mlops.dashboard.shap_labels import humanize_shap_feature_name
from churn_mlops.serving.api.schemas import CustomerFeatures

st.set_page_config(page_title="Churn MLOps", layout="wide")

dashboard_config = load_yaml_config("configs/dashboard.yaml")
API_BASE_URL = os.environ.get("CHURN_API_BASE_URL", dashboard_config["api_base_url"])
LOGO_PATH = Path(dashboard_config.get("logo_path", "assets/logo.png"))

_model_selection_dir = Path(
    load_yaml_config(dashboard_config["report_configs"]["model_selection"])[
        "reports_dir"
    ]
)
_monitoring_dir = Path(
    load_yaml_config(dashboard_config["report_configs"]["monitoring"])["reports_dir"]
)
_registry_dir = Path(
    load_yaml_config(dashboard_config["report_configs"]["registry"])["reports_dir"]
)

_BUSINESS_VALUE_CONFIG = dashboard_config.get("business_value", {})
RETENTION_CAMPAIGN_COST = float(
    _BUSINESS_VALUE_CONFIG.get("retention_campaign_cost_usd", 0.0)
)
CONTRACT_TERM_MONTHS: dict[str, int] = _BUSINESS_VALUE_CONFIG.get(
    "contract_term_months", {}
)

# Par divergente validado (skill de dataviz del proyecto): rojo = aumenta
# riesgo de churn, azul = lo reduce. Variantes claro/oscuro porque son hex
# fijos, no tokens CSS — Streamlit no expone el tema activo como CSS acá.
_DIVERGING_COLORS = {
    "dark": {"increase": "#e66767", "decrease": "#3987e5"},
    "light": {"increase": "#e34948", "decrease": "#2a78d6"},
}


def _run_module(module: str) -> subprocess.CompletedProcess[str]:
    """Corre un entrypoint manual de Fase 5/6 (`uv run python -m <module>`) y
    devuelve stdout/stderr — mismo comando que el usuario correría a mano."""
    return subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _extract_narrative(markdown_text: str) -> str:
    """Texto ANTES de la primera fila de tabla markdown (línea que empieza
    con "|") — la tabla se muestra aparte como st.dataframe, nunca las dos
    (ADR 0013: no duplicar la misma info en markdown y dataframe)."""
    narrative_lines = []
    for line in markdown_text.splitlines():
        if line.strip().startswith("|"):
            break
        narrative_lines.append(line)
    return "\n".join(narrative_lines).strip()


def _theme_colors() -> dict[str, str]:
    base = st.get_option("theme.base") or "dark"
    return _DIVERGING_COLORS["light"] if base == "light" else _DIVERGING_COLORS["dark"]


# ---------------------------------------------------------------- Predicción


def _init_prediction_state(specs: list[FieldSpec]) -> None:
    for spec in specs:
        st.session_state.setdefault(pf.field_key(spec.name), pf.default_value(spec))


def _randomize_prediction_state(specs: list[FieldSpec]) -> None:
    rng = random.Random()
    for spec in specs:
        st.session_state[pf.field_key(spec.name)] = pf.random_value(spec, rng)


def _read_prediction_state(specs: list[FieldSpec]) -> dict[str, object]:
    return {spec.name: st.session_state[pf.field_key(spec.name)] for spec in specs}


def _write_prediction_state(values: dict[str, object]) -> None:
    for name, value in values.items():
        st.session_state[pf.field_key(name)] = value


def _render_prediction_form(
    specs: list[FieldSpec],
    values: dict[str, object],
    locked: set[str],
    errors_by_field: dict[str, str],
) -> None:
    spec_by_name = {s.name: s for s in specs}
    for group_name, field_names in pf.FIELD_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "Datos personales")):
            columns = st.columns(3)
            for i, name in enumerate(field_names):
                spec = spec_by_name[name]
                widget = columns[i % 3]
                key = pf.field_key(name)
                disabled = name in locked

                if spec.kind == "select":
                    choices = (
                        [pf.NONE_SENTINEL, *spec.choices]
                        if spec.nullable
                        else list(spec.choices)
                    )
                    widget.selectbox(spec.label, choices, key=key, disabled=disabled)
                elif spec.kind == "int":
                    widget.number_input(
                        spec.label,
                        step=1,
                        key=key,
                        disabled=disabled,
                        min_value=pf.numeric_min_value(name, values),
                    )
                elif spec.kind == "float":
                    widget.number_input(
                        spec.label,
                        key=key,
                        disabled=disabled,
                        min_value=pf.numeric_min_value(name, values),
                    )
                else:
                    widget.text_input(spec.label, key=key, disabled=disabled)

                if name in errors_by_field:
                    widget.caption(f":red[⚠ {errors_by_field[name]}]")


def _render_prediction_result(
    payload: dict[str, object], known_columns: list[str]
) -> None:
    try:
        explanation = api_client.explain(API_BASE_URL, payload)
    except Exception as exc:
        st.error(f"No se pudo contactar la API en {API_BASE_URL}: {exc}")
        return

    proba = explanation["churn_probability"]
    card_1, card_2, card_3 = st.columns(3)
    card_1.metric("Probabilidad de churn", f"{proba:.1%}")
    card_2.metric(
        "Predicción", "Churn" if explanation["churn_prediction"] else "No churn"
    )
    card_3.metric("Champion version", explanation["champion_version"])

    contract = payload.get("contract")
    tenure = payload.get("tenure_in_months")
    monthly_charge = payload.get("monthly_charge")
    if (
        isinstance(contract, str)
        and contract in CONTRACT_TERM_MONTHS
        and isinstance(tenure, int | float)
        and isinstance(monthly_charge, int | float)
    ):
        retention = bv.estimate_retention_value(
            contract=contract,
            tenure_in_months=int(tenure),
            monthly_charge=float(monthly_charge),
            churn_probability=proba,
            contract_term_months=CONTRACT_TERM_MONTHS,
            retention_campaign_cost=RETENTION_CAMPAIGN_COST,
        )
        st.subheader("💰 ¿Vale la pena retener a este cliente?")
        bv_1, bv_2, bv_3 = st.columns(3)
        bv_1.metric(
            "Valor en riesgo",
            f"${retention.value_at_risk:,.2f}",
            help=f"Monthly Charge × {retention.remaining_months} mes(es) "
            "restantes del contrato actual",
        )
        bv_2.metric(
            "Pérdida esperada",
            f"${retention.expected_loss:,.2f}",
            help="Probabilidad de churn × valor en riesgo",
        )
        bv_3.metric("Costo de campaña de retención", f"${RETENTION_CAMPAIGN_COST:,.2f}")
        if retention.worth_retaining:
            st.success(
                f"Conviene intentar retener: beneficio neto esperado de "
                f"${retention.net_benefit:,.2f} (pérdida esperada − costo de "
                "retención)."
            )
        else:
            st.warning(
                f"No conviene retener con este costo de campaña: beneficio "
                f"neto esperado de ${retention.net_benefit:,.2f}."
            )

    shap_df = pd.DataFrame(
        [
            {"feature": humanize_shap_feature_name(name, known_columns), "value": value}
            for name, value in explanation["shap_values"].items()
        ]
    )
    shap_df["abs_value"] = shap_df["value"].abs()
    shap_df = shap_df.sort_values("abs_value", ascending=False).head(15)
    shap_df["direction"] = shap_df["value"].apply(
        lambda v: "Aumenta riesgo de churn" if v > 0 else "Reduce riesgo de churn"
    )
    colors = _theme_colors()

    st.subheader("Features con mayor impacto (SHAP, top 15)")
    chart = (
        alt.Chart(shap_df)
        .mark_bar(cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X("value:Q", title="Impacto SHAP (log-odds)"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Aumenta riesgo de churn", "Reduce riesgo de churn"],
                    range=[colors["increase"], colors["decrease"]],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value:Q", title="Impacto", format=".3f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


def render_prediction_page() -> None:
    st.title("🔮 Predicción de churn")

    city_choices = load_known_cities()
    specs = pf.build_specs(city_choices)
    _init_prediction_state(specs)

    if st.button(
        "🎲 Generar valores aleatorios",
        help="Rellena el formulario con un cliente al azar (rangos reales "
        "del dataset) — después podés editar cualquier campo",
    ):
        _randomize_prediction_state(specs)

    values = pf.apply_dependency_rules(_read_prediction_state(specs))
    _write_prediction_state(values)
    locked = pf.locked_fields(values)

    payload = pf.to_payload(values)
    try:
        CustomerFeatures.model_validate(payload)
        errors_by_field: dict[str, str] = {}
    except ValidationError as exc:
        errors_by_field = {str(err["loc"][0]): err["msg"] for err in exc.errors()}

    _render_prediction_form(specs, values, locked, errors_by_field)

    if errors_by_field:
        st.caption(":red[Corregí los campos marcados en rojo antes de predecir.]")

    if st.button("🔮 Predecir", type="primary", disabled=bool(errors_by_field)):
        known_columns = [s.label for s in specs]
        _render_prediction_result(payload, known_columns)


# ---------------------------------------------------------------- Producción


def render_production_page() -> None:
    st.title("📊 Modelo en producción")
    try:
        info = api_client.model_info(API_BASE_URL)
        card_1, card_2 = st.columns(2)
        card_1.metric("Champion version", info["champion_version"] or "sin champion")
        card_2.metric("Alias", info["champion_alias"])
        st.caption(info["registered_model_name"])
    except Exception as exc:
        st.warning(f"No se pudo leer /model-info de la API: {exc}")

    st.subheader("Historial de promoción champion/challenger")
    summary = read_text_report(_registry_dir / "promotion_summary.md")
    if summary:
        st.markdown(_extract_narrative(summary))
    else:
        st.info("Todavía no corrió run_promotion.py.")
    results = read_csv_report(_registry_dir / "promotion_results.csv")
    if results is not None:
        st.dataframe(results, use_container_width=True)

    st.subheader("Selección de modelo (Fase 3)")
    selection_summary = read_text_report(
        _model_selection_dir / "model_selection_summary.md"
    )
    if selection_summary:
        st.markdown(_extract_narrative(selection_summary))
    selection_results = read_csv_report(
        _model_selection_dir / "model_selection_results.csv"
    )
    if selection_results is not None:
        st.dataframe(selection_results, use_container_width=True)


# ---------------------------------------------------------------------- Drift


def render_drift_page() -> None:
    st.title("📈 Monitoreo de drift")
    results = read_csv_report(_monitoring_dir / "drift_monitoring_results.csv")
    if results is None:
        st.info("Todavía no corrió run_monitoring.py.")
        return

    st.subheader("Share de columnas con drift por batch")
    st.bar_chart(results.set_index("batch_name")["dataset_drift_share"])
    st.dataframe(results, use_container_width=True)


# ----------------------------------------------------------------- Operaciones


def render_operations_page() -> None:
    st.title("⚙️ Operaciones / Simulación")
    st.warning(
        "Estos botones corren los mismos scripts manuales de Fase 5/6 sobre "
        "esta máquina. Reentrenar/promover sobreescribe archivos locales "
        "(models/lightgbm_pipeline.pkl, models/champion_pipeline.pkl) — para "
        "que el resultado persista más allá de esta sesión hace falta correr "
        "`dvc add`/`dvc push` a mano después (mismo procedimiento que ADR 0012 "
        "documenta para correr estos scripts desde la terminal)."
    )

    st.subheader("1. Simular batches + detectar drift")
    if st.button("Correr detección de drift", icon="🔍"):
        with st.status(
            "Corriendo monitoring.run_monitoring...", expanded=True
        ) as status:
            result = _run_module("churn_mlops.monitoring.run_monitoring")
            st.code(result.stdout or result.stderr)
            status.update(label="Detección de drift completa", state="complete")
        st.toast("Drift detectado y reportado", icon="✅")

    st.subheader("2. Chequear si el drift amerita reentrenar")
    if st.button("Chequear trigger de reentrenamiento", icon="🧭"):
        with st.status(
            "Corriendo monitoring.run_retrain_check...", expanded=True
        ) as status:
            result = _run_module("churn_mlops.monitoring.run_retrain_check")
            st.code(result.stdout or result.stderr)
            status.update(label="Chequeo completo", state="complete")
        st.toast("Trigger evaluado", icon="✅")

    st.subheader("3. Reentrenar el challenger")
    confirm_retrain = st.checkbox(
        "Confirmo que quiero reentrenar y sobreescribir models/lightgbm_pipeline.pkl"
    )
    if st.button("Reentrenar challenger", icon="🔄", disabled=not confirm_retrain):
        with st.status("Corriendo training.run_retrain...", expanded=True) as status:
            result = _run_module("churn_mlops.training.run_retrain")
            st.code(result.stdout or result.stderr)
            status.update(label="Reentrenamiento completo", state="complete")
        st.toast("Challenger reentrenado", icon="✅")

    st.subheader("4. Comparar y promover challenger → champion")
    confirm_promote = st.checkbox(
        "Confirmo que quiero comparar y, si corresponde, promover "
        "(sobreescribe models/champion_pipeline.pkl)"
    )
    if st.button("Comparar y promover", icon="🚀", disabled=not confirm_promote):
        with st.status("Corriendo registry.run_promotion...", expanded=True) as status:
            result = _run_module("churn_mlops.registry.run_promotion")
            st.code(result.stdout or result.stderr)
            status.update(label="Comparación/promoción completa", state="complete")
        st.toast("Comparación de champion/challenger completa", icon="✅")


# -------------------------------------------------------------------- Entrypoint


def main() -> None:
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))

    pages = [
        st.Page(render_prediction_page, title="Predicción", icon="🔮", default=True),
        st.Page(render_production_page, title="Producción", icon="📊"),
        st.Page(render_drift_page, title="Drift", icon="📈"),
        st.Page(render_operations_page, title="Operaciones", icon="⚙️"),
    ]
    st.navigation(pages, position="sidebar").run()


main()
