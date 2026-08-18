"""Genera los graficos de datos reales embebidos en docs/CASE_STUDY.md
(comparacion de familias de modelo, churn por tipo de contrato) a partir de
reports/ y data/raw/telco.csv. Se corre a mano cuando esos datos cambian,
mismo patron que dashboard/export_known_cities.py o monitoring/run_monitoring.py
(no es un stage de dvc.yaml):

    uv run python scripts/generate_case_study_charts.py

Paleta y specs de marca siguiendo la skill de dataviz del proyecto (bar
"emphasis" para destacar un ganador, secuencial ordinal para una categoria
con orden natural) -- ver docs/CASE_STUDY.md para donde se usa cada grafico."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "sans-serif"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"
DEEMPHASIS = "#c3c2b7"


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", colors=MUTED, labelsize=10.5, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Chart 1: comparacion de familias de modelo por F1 (emphasis: LightGBM)
# ---------------------------------------------------------------------------
model_df = pd.read_csv("reports/model_selection_results.csv")
name_map = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "logistic_regression": "Regresión Logística",
}
model_df["label"] = model_df["family"].map(name_map)
model_df = model_df.sort_values("f1", ascending=True)

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=180)
colors = [ACCENT if fam == "lightgbm" else DEEMPHASIS for fam in model_df["family"]]
bars = ax.barh(model_df["label"], model_df["f1"], color=colors, height=0.55, zorder=3)

for bar, value, fam in zip(bars, model_df["f1"], model_df["family"], strict=True):
    weight = "bold" if fam == "lightgbm" else "normal"
    color = INK if fam == "lightgbm" else INK_SECONDARY
    ax.text(
        value + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        ha="left",
        fontsize=11,
        fontweight=weight,
        color=color,
    )

ax.set_xlim(0.85, 0.97)
ax.set_xlabel("F1-Score (test set)", color=MUTED, fontsize=10.5)
_style_ax(ax)
ax.set_title(
    "LightGBM gana la comparación de 4 familias — F1-Score",
    loc="left",
    fontsize=13,
    color=INK,
    fontweight="bold",
    pad=14,
)
fig.text(
    0.01,
    -0.02,
    "Fase 3 — mismo protocolo de evaluación (train/test split, features idénticas) para las 4 familias.",
    fontsize=9,
    color=MUTED,
)
fig.tight_layout()
fig.savefig(
    "docs/assets/case-study/model-comparison.png",
    bbox_inches="tight",
    facecolor=SURFACE,
)
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 2: churn rate por tipo de contrato (secuencial ordinal)
# ---------------------------------------------------------------------------
raw_df = pd.read_csv("data/raw/telco.csv")
churn_by_contract = (
    raw_df.groupby("Contract")["Churn Label"]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .reindex(["Month-to-Month", "One Year", "Two Year"])
)
counts = (
    raw_df.groupby("Contract")
    .size()
    .reindex(["Month-to-Month", "One Year", "Two Year"])
)

seq_colors = [
    "#104281",
    "#3987e5",
    "#86b6ef",
]  # steps 650 / 400 / 250, mas oscuro = mas churn

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=180)
bars = ax.barh(
    churn_by_contract.index[::-1],
    churn_by_contract.values[::-1],
    color=seq_colors[::-1],
    height=0.55,
    zorder=3,
)

for bar, value, n in zip(
    bars, churn_by_contract.values[::-1], counts.values[::-1], strict=True
):
    ax.text(
        value + 1.0,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%  (n={n:,})",
        va="center",
        ha="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )

ax.set_xlim(0, 58)
ax.set_xlabel("% de clientes que hicieron churn", color=MUTED, fontsize=10.5)
_style_ax(ax)
ax.set_title(
    "A menor compromiso contractual, más churn",
    loc="left",
    fontsize=13,
    color=INK,
    fontweight="bold",
    pad=14,
)
fig.text(
    0.01,
    -0.02,
    "Dataset completo (7,043 clientes) — Month-to-Month churna 18× más que Two Year.",
    fontsize=9,
    color=MUTED,
)
fig.tight_layout()
fig.savefig(
    "docs/assets/case-study/churn-by-contract.png",
    bbox_inches="tight",
    facecolor=SURFACE,
)
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 3: diagrama de arquitectura (capas, no es un chart de datos -- solo
# reusa los tokens de color de la paleta para que las 3 imagenes combinen)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.8, 4.4), dpi=180)
ax.set_xlim(0, 1180)
ax.set_ylim(-45, 420)
ax.invert_yaxis()
ax.axis("off")
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

boxes = [
    (24, 60, 180, 92, "Datos", "Pandera · DVC", "validación + versionado", False),
    (
        244,
        60,
        180,
        92,
        "Features",
        "build_features.py",
        "solo ingeniería, sin encoding",
        False,
    ),
    (464, 60, 180, 92, "Training", "Optuna · MLflow", "tuning + tracking", False),
    (
        684,
        60,
        210,
        92,
        "Registry / Serving",
        "MLflow Registry · FastAPI",
        "champion / challenger",
        True,
    ),
    (
        934,
        60,
        222,
        92,
        "Monitoreo",
        "Evidently AI",
        "drift + trigger de reentreno",
        False,
    ),
    (
        614,
        250,
        350,
        100,
        "Presentación",
        "FastAPI (predicción + SHAP)  +  Streamlit",
        "incluye el valor de negocio por cliente",
        True,
    ),
]

for x, y, w, h, title, tool, sub, emphasis in boxes:
    color = ACCENT if emphasis else BASELINE
    lw = 2.2 if emphasis else 1.4
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=10",
        linewidth=lw,
        edgecolor=color,
        facecolor=SURFACE,
        zorder=3,
    )
    ax.add_patch(box)
    cx = x + w / 2
    ax.text(
        cx,
        y + 32,
        title,
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        cx, y + 56, tool, ha="center", va="center", fontsize=10.5, color=INK_SECONDARY
    )
    ax.text(cx, y + 76, sub, ha="center", va="center", fontsize=9.5, color=MUTED)

arrow_style = dict(
    arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=MUTED, zorder=2
)
for x0, x1 in [(204, 244), (424, 464), (644, 684), (894, 934)]:
    ax.add_patch(FancyArrowPatch((x0, 106), (x1, 106), **arrow_style))

# flecha vertical: Registry/Serving -> Presentacion
ax.add_patch(
    FancyArrowPatch(
        (789, 152),
        (789, 250),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=ACCENT,
        zorder=2,
    )
)

# loop de reentreno: Monitoreo -> Training, ruta explicita en escuadra por
# arriba de toda la fila (un solo arc3 no despejaba las cajas de forma
# confiable) -- sube desde Monitoreo, cruza, y baja a Training.
elbow = Path(
    [(1045, 58), (1045, -22), (554, -22), (554, 58)],
    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO],
)
loop = FancyArrowPatch(
    path=elbow,
    arrowstyle="-|>",
    mutation_scale=12,
    linewidth=1.3,
    linestyle=(0, (4, 3)),
    color=BASELINE,
    zorder=1,
)
ax.add_patch(loop)
ax.text(
    800,
    -32,
    "si el drift amerita: reentrenar (manual, ADR 0012)",
    ha="center",
    fontsize=9.5,
    color=MUTED,
    style="italic",
)

ax.text(
    24,
    210,
    "Cada capa solo conoce a la anterior — servir predicciones nunca depende de que el monitoreo esté arriba.",
    fontsize=10.5,
    color=INK_SECONDARY,
)
ax.text(
    24,
    400,
    "14 ADRs documentan las decisiones detrás de cada flecha — ver docs/decisions/.",
    fontsize=9.5,
    color=MUTED,
)

fig.tight_layout()
fig.savefig(
    "docs/assets/case-study/architecture.png", bbox_inches="tight", facecolor=SURFACE
)
plt.close(fig)

print("OK — 3 charts generated")
print(churn_by_contract)
