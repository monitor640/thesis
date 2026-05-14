"""Figures for thesis modeling (holdout prediction distributions)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import colormaps
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from ridgeplot import ridgeplot

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
DEFAULT_HOLDOUT_RIDGE_PATH = FIGURES_DIR / "holdout_ridgelines.png"
MODELING_TABLE_PATH = DATA_DIR / "modeling_table.csv"
UKU_LABELING_PATH = DATA_DIR / "uku-labeling.csv"
SIMON_LABELING_PATH = DATA_DIR / "simon-labeling.csv"
DEFAULT_LABEL_BARS_PATH = FIGURES_DIR / "propaganda_label_bars.png"
DEFAULT_HOLDOUT_SCATTER_PATH = FIGURES_DIR / "holdout_pred_vs_true_scatter.png"
SKLEARN_IMPORTANCES_JSON = ROOT / "checkpoints" / "modeling" / "sklearn_feature_importances.json"
DEFAULT_FEATURE_IMPORTANCE_SCATTER_PATH = FIGURES_DIR / "feature_importance_scatter.png"

# Facet layout (2×3): row1 linear | rf | bert; row2 llm | llm_with_features | empty.
MODEL_ORDER_SCATTER = ("linear", "rf", "bert", "llm", "llm_with_features")

# Consensus uses half-step scores; single annotators use integers 0–4.
CONSENSUS_SCORE_ORDER = [i * 0.5 for i in range(9)]
INTEGER_SCORE_ORDER = [0, 1, 2, 3, 4]

HOLDOUT_CSVS = [
    DATA_DIR / "holdout_llm_predictions.csv",
    DATA_DIR / "holdout_bert_predictions.csv",
    DATA_DIR / "holdout_sklearn_predictions.csv",
    DATA_DIR / "holdout_llm_with_features_predictions.csv",
]

NAME_MAP = {
    "llm": "GPT-5.4-nano",
    "bert": "XLM-RoBERTa",
    "linear": "Linear Regression",
    "rf": "Random Forest",
    "llm_with_features": "GPT-5.4-nano (with features)",
    "y_true": "Annotator consensus",
}

# Bottom → top rows in the ridgeline plot.
RIDGE_ORDER = (
    "y_true",
    "llm",
    "bert",
    "linear",
    "rf",
    "llm_with_features"
)


def pred_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pred_")]


def load_holdout_predictions_wide() -> pd.DataFrame:
    """Merge holdout CSVs on ``channel`` + ``telegram_id`` → ``y_true``, ``pred_<model>``."""
    master = pd.read_csv(HOLDOUT_CSVS[0])[["channel", "telegram_id", "y_true"]].copy()
    merge_cols = ["channel", "telegram_id"]
    for path in HOLDOUT_CSVS:
        df = pd.read_csv(path)
        for col in pred_columns(df):
            key = col.removeprefix("pred_")
            piece = df[merge_cols + [col]].rename(columns={col: f"pred_{key}"})
            master = master.merge(piece, on=merge_cols, how="inner")
    return master


def _ordered_series(names: list[str]) -> list[str]:
    rank = {n: i for i, n in enumerate(RIDGE_ORDER)}
    return sorted(names, key=lambda s: (rank.get(s, 10_000), s))


def holdout_scores_long() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    first = pd.read_csv(HOLDOUT_CSVS[0])
    for v in first["y_true"].astype(float):
        rows.append({"series": "y_true", "value": v})

    for path in HOLDOUT_CSVS:
        df = pd.read_csv(path)
        for col in pred_columns(df):
            label = col.removeprefix("pred_")
            for v in df[col].astype(float):
                rows.append({"series": label, "value": v})

    return pd.DataFrame(rows)


def plot_holdout_ridgelines(
    out_path: Path | None = None,
    *,
    width: int = 920,
    row_height_px: int = 72,
) -> None:
    """Ridgeline densities via [ridgeplot](https://ridgeplot.readthedocs.io/) (Plotly)."""
    long_df = holdout_scores_long()
    categories = _ordered_series(long_df["series"].unique().tolist())

    samples = [
        long_df.loc[long_df["series"] == cat, "value"].astype(float).values
        for cat in categories
    ]

    vals = long_df["value"].astype(float).values
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.03 * (hi - lo if hi > lo else 1.0)
    kde_points = np.linspace(lo - pad, hi + pad, 400)

    fig = ridgeplot(
        samples=samples,
        labels=categories,
        colorscale="viridis",
        colormode="row-index",
        opacity=0.85,
        kde_points=kde_points,
        spacing=0.55,
        xpad=0.05,
        line_color="fill-color",
    )
    fig.update_layout(
        height=max(480, row_height_px * len(categories)),
        width=width,
        font_size=13,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Score",
        yaxis_title="",
        showlegend=False,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out_path), scale=2)


def plot_holdout_pred_vs_true_facets(
    out_path: Path | None = None,
    *,
    jitter: float = 0.06,
    point_alpha: float = 0.4,
    rng_seed: int = 42,
    figsize: tuple[float, float] = (11.5, 7.8),
    show: bool = True,
) -> None:
    """Faceted scatter: true vs predicted (holdout); y=x; jittered x. Facet order: sklearn, BERT, LLMs."""
    df = load_holdout_predictions_wide()
    yt = df["y_true"].astype(float).to_numpy()

    pred_cols = [f"pred_{k}" for k in MODEL_ORDER_SCATTER]
    all_vals = np.concatenate([yt] + [df[c].astype(float).to_numpy() for c in pred_cols])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    span = vmax - vmin if vmax > vmin else 1.0
    pad = 0.06 * span
    lim_lo, lim_hi = vmin - pad, vmax + pad

    rng = np.random.default_rng(rng_seed)
    x_jitter = yt + rng.normal(0.0, jitter, size=len(yt))

    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    axes_flat = axes.flatten()

    diagonal_color = "#c0392b"
    point_color = "#1f77b4"

    for i, key in enumerate(MODEL_ORDER_SCATTER):
        ax = axes_flat[i]
        yp = df[f"pred_{key}"].astype(float).to_numpy()

        ax.scatter(
            x_jitter,
            yp,
            alpha=point_alpha,
            s=18,
            c=point_color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

        ax.plot(
            [lim_lo, lim_hi],
            [lim_lo, lim_hi],
            color=diagonal_color,
            ls="--",
            lw=1.4,
            zorder=1,
        )

        ax.set_title(NAME_MAP.get(key, key))
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.28, ls=":", linewidth=0.8)

    axes_flat[5].axis("off")

    fig.supxlabel("True score", fontsize=12)
    fig.supylabel("Predicted score", fontsize=12)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def _count_at_scores(series: pd.Series, scores: list[float]) -> list[int]:
    """Count rows matching each score (float-safe)."""
    arr = series.dropna().astype(float).to_numpy()
    out: list[int] = []
    for v in scores:
        out.append(int(np.sum(np.isclose(arr, float(v), rtol=0.0, atol=1e-9))))
    return out


def _score_xtick_labels(scores: list[float]) -> list[str]:
    labels: list[str] = []
    for v in scores:
        if abs(v - round(v)) < 1e-9:
            labels.append(str(int(round(v))))
        else:
            labels.append(str(v))
    return labels


def plot_propaganda_label_bars(
    out_path: Path | None = None,
    *,
    figsize: tuple[float, float] = (14.5, 4.8),
    show: bool = True,
) -> None:
    """Three panels: two annotators (0–4), then consensus mean (9 half steps). Counts per bin."""
    consensus = pd.read_csv(MODELING_TABLE_PATH, usecols=["propa_consensus"])["propa_consensus"]
    uku = pd.read_csv(UKU_LABELING_PATH, usecols=["Propa"])["Propa"]
    simon = pd.read_csv(SIMON_LABELING_PATH, usecols=["Propa"])["Propa"]

    float_ints = [float(x) for x in INTEGER_SCORE_ORDER]
    heights_cons = _count_at_scores(consensus, CONSENSUS_SCORE_ORDER)
    heights_uku = _count_at_scores(uku.astype(float), float_ints)
    heights_sim = _count_at_scores(simon.astype(float), float_ints)

    # Left to right: annotators, then consensus (rightmost).
    panels: list[tuple[str, list[float], list[int], bool]] = [
        ("Annotator 1", INTEGER_SCORE_ORDER, heights_uku, False),
        ("Annotator 2", INTEGER_SCORE_ORDER, heights_sim, False),
        ("Consensus mean", CONSENSUS_SCORE_ORDER, heights_cons, True),
    ]

    ymax_global = max((max(h, default=0) for _, _, h, _ in panels), default=1)
    pad = max(ymax_global * 0.03, 3)
    y_top = ymax_global + pad * 4

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    cmap = colormaps["Blues"]

    for ax, (title, scores, heights, rotate_x) in zip(axes, panels, strict=True):
        n = len(scores)
        x = np.arange(n, dtype=float)
        colors = [cmap(0.28 + 0.68 * (i / max(n - 1, 1))) for i in range(n)]
        ax.bar(x, heights, color=colors, edgecolor="white", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(_score_xtick_labels(scores), rotation=45 if rotate_x else 0, ha="right" if rotate_x else "center")
        ax.set_title(title)
        ax.set_xlabel("Propaganda score")
        ax.set_ylim(0, y_top)

        for xi, h in zip(x, heights, strict=True):
            ax.text(xi, h + pad, str(h), ha="center", va="bottom", fontsize=10)

    axes[0].set_ylabel("Count")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_feature_importance_scatter(
    importances_json: Path | None = None,
    out_path: Path | None = None,
    figsize: tuple[float, float] = (10, 7),
) -> None:
    if importances_json is None:
        importances_json = SKLEARN_IMPORTANCES_JSON 
    with open(importances_json, encoding="utf-8") as f:
        data = json.load(f)

    features = data["features"]
    lr_coefs = np.asarray(data["linear"]["coef_on_scaled_features"], dtype=float)
    rf_imp = np.asarray(data["rf"]["feature_importances"], dtype=float)

    abs_coefs = np.abs(lr_coefs)
    colors = ["#2c7bb6" if c >= 0 else "#d7191c" for c in lr_coefs]

    fig, ax = plt.subplots(figsize=figsize)

    # Set axes generously so labels fit
    x_max = float(abs_coefs.max() * 1.15)
    y_max = float(rf_imp.max() * 1.15)
    ax.set_xlim(-0.005, x_max)
    ax.set_ylim(-0.005, y_max)

    # Plot dots
    ax.scatter(abs_coefs, rf_imp, c=colors, s=70, alpha=0.85,
               edgecolors="black", linewidths=0.5, zorder=3)

    manual_offsets = {
    "caps_ratio":  (0.004, 0.008),   # default: a bit up and right
    "verb_count":  (-0.006, -0.008),  # bumped slightly down to clear caps_ratio
    }

    for x, y, name in zip(abs_coefs, rf_imp, features):
        # Default offset
        ha = "left"
        offset_x, offset_y = 0.006, 0.0
        
        # Apply manual override if present
        if name in manual_offsets:
            offset_x, offset_y = manual_offsets[name]
        elif x > x_max * 0.65:
            ha = "right"
            offset_x = -0.006
        
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(x + offset_x, y + offset_y),
            ha=ha,
            va="center",
            fontsize=9,
        )

    # Light grid
    ax.grid(True, alpha=0.3)

    # Axis labels
    ax.set_xlabel("|Linear regression coefficient| (standardised features)")
    ax.set_ylabel("Random Forest feature importance")
    ax.set_title("Feature importance: linear vs. tree-based models")

    # Legend (top right but inside, with padding)
    legend_handles = [
        Patch(facecolor="#2c7bb6", edgecolor="black", linewidth=0.5,
              label="Positive LR coefficient"),
        Patch(facecolor="#d7191c", edgecolor="black", linewidth=0.5,
              label="Negative LR coefficient"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=9)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_holdout_ridgelines(out_path=DEFAULT_HOLDOUT_RIDGE_PATH)
    plot_propaganda_label_bars(out_path=DEFAULT_LABEL_BARS_PATH)
    plot_holdout_pred_vs_true_facets(out_path=DEFAULT_HOLDOUT_SCATTER_PATH)
    if SKLEARN_IMPORTANCES_JSON.is_file():
        plot_feature_importance_scatter(out_path=DEFAULT_FEATURE_IMPORTANCE_SCATTER_PATH)
    else:
        print(
            f"Skipping feature importance scatter (missing {SKLEARN_IMPORTANCES_JSON})",
            flush=True,
        )
