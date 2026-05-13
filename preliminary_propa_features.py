"""
Exploratory comparison of numeric NLP features by propaganda label (Propa).

Uses rows labeled ``propa`` vs ``ei ole propa`` only. Unlabeled, ``üle vaadata``,
and other values are reported in counts but excluded from contrast plots/tests.

Usage:
  python preliminary_propa_features.py

Paths and outputs: ``config.py`` (LABELING_WITH_FEATURES_CSV, FIGURES_DIR, PRELIMINARY_*_CSV).

Also writes subset figures (links, dependency syntax, simple text, style lexicon):
  propa_link_*.png, propa_dep_syntax_*.png,
  propa_simple_text_*.png, propa_bw_and_fullcaps_*.png

Mann–Whitney + Benjamini–Hochberg FDR vs ``not_propaganda`` (default CSV:
  data/preliminary_propa_group_tests.csv); includes sklearn ``mutual_info_classif`` scores
  (per-column ``discrete_features`` from ``config.MUTUAL_INFO_CONTINUOUS_FEATURE_COLS`` so
  sparse counts are not all estimated as k-NN continuous, which floors MI at 0) to rank
  with Cohen's d. Cohen's d figure marks FDR-significant *.

``modeling.feature_config`` applies the thesis OR-rule (|d|, BH q, MI with gates) when
selecting which of these columns enter sklearn / LLM-with-features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    BW_AND_FULLCAPS_COLS,
    DEP_SYNTAX_FEATURE_COLS,
    EMOJI_FEATURE_COLS,
    FEATURE_COLS,
    FIGURES_DIR,
    LABELING_WITH_FEATURES_CSV,
    LINK_FEATURE_COLS,
    MUTUAL_INFO_CONTINUOUS_FEATURE_COLS,
    PRELIMINARY_FDR_ALPHA,
    PRELIMINARY_FEATURE_SUMMARY_CSV,
    PRELIMINARY_MI_RANDOM_STATE,
    PRELIMINARY_TESTS_CSV,
    SIMPLE_TEXT_FEATURE_COLS,
)


def propa_group_label(x) -> str:
    if pd.isna(x):
        return "unlabeled"
    s = str(x).strip().lower()
    if "ei ole propa" in s:
        return "not_propaganda"
    if "üle vaadata" in s or "ule vaadata" in s:
        return "review"
    if s == "propa":
        return "propaganda"
    return "other"


def plot_propa_label_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    """How many rows in each Propa bucket (full sample)."""
    order = ["propaganda", "not_propaganda", "review", "unlabeled", "other"]
    vc = df["propa_group"].value_counts()
    idx = [k for k in order if k in vc.index and vc[k] > 0]
    counts = vc.reindex(idx).fillna(0).astype(int)
    if counts.empty:
        print("No Propa groups to plot for label counts.")
        return out_dir / "_skipped_label_counts.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette("Set2", n_colors=max(len(counts), 1))
    ax.barh(counts.index.astype(str), counts.values, color=palette[: len(counts)])
    ax.set_xlabel("Posts")
    ax.set_ylabel("")
    ax.set_title("Labeling progress by Propa bucket")
    pad = max(counts.max(), 1) * 0.02
    for i, v in enumerate(counts.values):
        ax.text(v + pad, i, str(v), va="center", fontsize=10)
    ax.margins(x=0.08)
    fig.tight_layout()
    path = out_dir / "propa_label_counts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_mean_bars(summary_long: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bars: mean of each feature for propaganda vs not_propaganda."""
    sub = summary_long[summary_long["propa_group"].isin(("propaganda", "not_propaganda"))]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(
        data=sub,
        x="feature",
        y="mean",
        hue="propa_group",
        order=FEATURE_COLS,
        ax=ax,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean")
    ax.set_title("Mean feature values: propaganda vs not propaganda (labeled rows)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax.legend(title="")
    fig.tight_layout()
    path = out_dir / "propa_feature_mean_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _cohens_d_per_feature(summary_long: pd.DataFrame) -> pd.Series:
    """
    Cohen's d (propaganda vs not_propaganda) using pooled SD from summary table.

    Comparable across features with different units. Positive => higher mean under propaganda.
    """
    sub = summary_long[
        summary_long["propa_group"].isin(("propaganda", "not_propaganda"))
    ]
    pv_mean = sub.pivot(index="feature", columns="propa_group", values="mean")
    pv_std = sub.pivot(index="feature", columns="propa_group", values="std")
    pv_n = sub.pivot(index="feature", columns="propa_group", values="n")
    if "propaganda" not in pv_mean.columns or "not_propaganda" not in pv_mean.columns:
        return pd.Series(dtype=float)

    out: dict[str, float] = {}
    for feat in FEATURE_COLS:
        if feat not in pv_mean.index:
            continue
        m_p = float(pv_mean.loc[feat, "propaganda"])
        m_n = float(pv_mean.loc[feat, "not_propaganda"])
        s_p = pv_std.loc[feat, "propaganda"]
        s_n = pv_std.loc[feat, "not_propaganda"]
        n_p = int(pv_n.loc[feat, "propaganda"])
        n_n = int(pv_n.loc[feat, "not_propaganda"])
        if n_p < 2 or n_n < 2:
            continue
        v_p = float(s_p) ** 2 if pd.notna(s_p) else 0.0
        v_n = float(s_n) ** 2 if pd.notna(s_n) else 0.0
        df_denom = n_p + n_n - 2
        if df_denom <= 0:
            continue
        pooled_var = ((n_p - 1) * v_p + (n_n - 1) * v_n) / df_denom
        if pooled_var <= 0 or np.isnan(pooled_var):
            if m_p == m_n:
                out[feat] = 0.0
            continue
        out[feat] = (m_p - m_n) / np.sqrt(pooled_var)
    return pd.Series(out)


def _benjamini_hochberg_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (FDR control)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    p_sorted = np.clip(p[order], 0.0, 1.0)
    adj_sorted = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = p_sorted[i] * n / rank
        running_min = min(val, running_min)
        adj_sorted[i] = running_min
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj_sorted, 0.0, 1.0)
    return out


def mann_whitney_tests_vs_propa(
    labeled: pd.DataFrame,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame | None:
    """
    Two-sided Mann–Whitney U: propaganda vs not_propaganda per feature.

    H0: distributions are identical; H1: distributions differ (location/shape).

    Adds raw p-values and Benjamini–Hochberg FDR-adjusted q-values (``p_fdr_bh``).
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return None

    rows: list[dict] = []
    for col in FEATURE_COLS:
        a = pd.to_numeric(
            labeled.loc[labeled["propa_group"] == "propaganda", col],
            errors="coerce",
        ).dropna()
        b = pd.to_numeric(
            labeled.loc[labeled["propa_group"] == "not_propaganda", col],
            errors="coerce",
        ).dropna()
        rec: dict = {"feature": col, "n_propa": len(a), "n_not": len(b)}
        if len(a) < 3 or len(b) < 3:
            rec["statistic_mw"] = np.nan
            rec["p_mannwhitney"] = np.nan
            rows.append(rec)
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rec["statistic_mw"] = float(stat)
        rec["p_mannwhitney"] = float(p)
        rows.append(rec)

    out = pd.DataFrame(rows)
    mask = out["p_mannwhitney"].notna()
    out["p_fdr_bh"] = np.nan
    if mask.any():
        out.loc[mask, "p_fdr_bh"] = _benjamini_hochberg_adjust(
            out.loc[mask, "p_mannwhitney"].values
        )
    out["sig_fdr"] = out["p_fdr_bh"] < fdr_alpha
    return out


def _mutual_info_discrete_mask(feature_cols: list[str]) -> np.ndarray:
    """
    True = treat column as discrete (integer counts, dep stats on small support).
    False = ratio / real-valued; sklearn uses k-NN continuous MI for that column only.
    """
    return np.fromiter(
        (c not in MUTUAL_INFO_CONTINUOUS_FEATURE_COLS for c in feature_cols),
        dtype=bool,
        count=len(feature_cols),
    )


def mutual_info_scores_vs_propa(
    labeled: pd.DataFrame,
    feature_cols: list[str],
    *,
    random_state: int = 0,
) -> tuple[pd.Series, int, np.ndarray] | None:
    """
    sklearn MI between each numeric feature and the binary class (``propaganda`` vs
    ``not_propaganda``). Uses complete rows only (any NaN in a feature row drops that row).

    Count-like columns use the discrete feature estimator; ratio columns use the continuous
    one (``config.MUTUAL_INFO_CONTINUOUS_FEATURE_COLS``). Ranks with Cohen's d; not calibrated
    across estimators, but reduces spurious all-zero scores from the k-NN path on counts.
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError:
        return None

    sub = labeled[labeled["propa_group"].isin(("propaganda", "not_propaganda"))].copy()
    y = (sub["propa_group"] == "propaganda").astype(np.int8).to_numpy()
    x_df = sub[feature_cols].apply(pd.to_numeric, errors="coerce")
    ok = x_df.notna().all(axis=1)
    n_ok = int(ok.sum())
    y_ok = y[ok.to_numpy()]
    if n_ok < 10 or y_ok.min() == y_ok.max():
        return None
    x_arr = x_df.loc[ok].to_numpy(dtype=np.float64)
    discrete = _mutual_info_discrete_mask(feature_cols)
    try:
        mi = mutual_info_classif(
            x_arr,
            y_ok,
            discrete_features=discrete,
            random_state=random_state,
        )
    except (ValueError, TypeError):
        return None
    return (
        pd.Series(mi, index=feature_cols, name="mutual_info_classif"),
        n_ok,
        discrete,
    )


def plot_cohens_d_bars(
    summary_long: pd.DataFrame,
    out_dir: Path,
    mw_tests: pd.DataFrame | None = None,
    fdr_alpha: float = 0.05,
) -> Path | None:
    """Horizontal bar: Cohen's d; optional * for FDR-significant Mann–Whitney tests."""
    d = _cohens_d_per_feature(summary_long).dropna()
    if d.empty:
        return None
    d = d.reindex(d.abs().sort_values(ascending=False).index)

    sig_map: dict[str, bool] = {}
    if mw_tests is not None and len(mw_tests):
        for _, r in mw_tests.iterrows():
            feat = r["feature"]
            if pd.notna(r.get("sig_fdr")) and bool(r["sig_fdr"]):
                sig_map[str(feat)] = True

    ylabels = []
    for feat in d.index:
        lab = str(feat).replace("_", " ")
        if sig_map.get(str(feat), False):
            lab = f"{lab} *"
        ylabels.append(lab)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c44e52" if v > 0 else "#55a868" for v in d.values]
    ax.barh(range(len(d)), d.values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (difference / pooled SD; positive => higher under propaganda)")
    ax.set_title("Standardized mean difference by feature (scale-free vs raw units)")
    if sig_map:
        fig.subplots_adjust(bottom=0.18)
        fig.text(
            0.5,
            0.04,
            f"* Mann-Whitney U (two-sided); FDR-BH q < {fdr_alpha}. "
            "H0: same distribution; H1: distributions differ.",
            ha="center",
            fontsize=8,
            style="italic",
        )
    else:
        fig.tight_layout()
    path = out_dir / "propa_feature_cohens_d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_subset_boxplot(
    melted: pd.DataFrame,
    feature_order: list[str],
    out_path: Path,
    title: str,
    figsize: tuple[float, float],
) -> Path | None:
    """Boxplot for a subset of features (same hue as global plots)."""
    sub = melted[melted["feature"].isin(feature_order)]
    if sub.empty or sub["value"].notna().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=sub,
        x="feature",
        y="value",
        hue="propa_group",
        order=feature_order,
        ax=ax,
        fliersize=2,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_link_mean_bars(summary_long: pd.DataFrame, out_dir: Path) -> Path | None:
    """Mean link totals, own-site, and other-site links (propaganda vs not)."""
    sub = summary_long[
        summary_long["feature"].isin(LINK_FEATURE_COLS)
        & summary_long["propa_group"].isin(("propaganda", "not_propaganda"))
    ]
    if len(sub) < len(LINK_FEATURE_COLS) * 2:
        return None
    label_map = dict(
        zip(
            LINK_FEATURE_COLS,
            ["All links", "Own site", "Other sites"],
        )
    )
    sub = sub.copy()
    sub["link_kind"] = sub["feature"].map(label_map)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        data=sub,
        x="link_kind",
        y="mean",
        hue="propa_group",
        order=["All links", "Own site", "Other sites"],
        ax=ax,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean (per post)")
    ax.set_title("Links: mean counts by label group")
    ax.legend(title="")
    fig.tight_layout()
    path = out_dir / "propa_link_means_by_propa.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_link_own_share_boxplot(labeled: pd.DataFrame, out_dir: Path) -> Path | None:
    """Among posts with at least one link: share of links pointing to first-party URLs."""
    sub = labeled[labeled["propa_group"].isin(("propaganda", "not_propaganda"))].copy()
    sub["_lc"] = pd.to_numeric(sub["link_count"], errors="coerce").fillna(0)
    sub["_own"] = pd.to_numeric(sub["link_own_site_count"], errors="coerce").fillna(0)
    sub = sub[sub["_lc"] > 0]
    if len(sub) < 8:
        return None
    sub["link_own_share"] = sub["_own"] / sub["_lc"]
    sub["propa_label"] = sub["propa_group"].map(
        {"propaganda": "Propaganda", "not_propaganda": "Not propaganda"}
    )
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.boxplot(
        data=sub,
        x="propa_label",
        y="link_own_share",
        hue="propa_label",
        order=["Propaganda", "Not propaganda"],
        hue_order=["Propaganda", "Not propaganda"],
        dodge=False,
        ax=ax,
        palette=["#4c78a8", "#f58518"],
        fliersize=2,
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Own links / all links")
    ax.set_title("Share of first-party links (posts with ≥1 link only)")
    fig.tight_layout()
    path = out_dir / "propa_link_own_share_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dep_syntax_mean_bars(summary_long: pd.DataFrame, out_dir: Path) -> Path | None:
    """Grouped bars: dependency syntax means only (readable rotation)."""
    sub = summary_long[
        summary_long["feature"].isin(DEP_SYNTAX_FEATURE_COLS)
        & summary_long["propa_group"].isin(("propaganda", "not_propaganda"))
    ]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=sub,
        x="feature",
        y="mean",
        hue="propa_group",
        order=DEP_SYNTAX_FEATURE_COLS,
        ax=ax,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean")
    ax.set_title("Dependency syntax (MaltParser): mean by label group")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=28, ha="right")
    ax.legend(title="")
    fig.tight_layout()
    path = out_dir / "propa_dep_syntax_means_by_propa.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_simple_text_mean_bars(summary_long: pd.DataFrame, out_dir: Path) -> Path | None:
    """Grouped bars: word/sentence lengths, punctuation counts, letter caps ratio."""
    sub = summary_long[
        summary_long["feature"].isin(SIMPLE_TEXT_FEATURE_COLS)
        & summary_long["propa_group"].isin(("propaganda", "not_propaganda"))
    ]
    if sub.empty:
        return None
    label_map = dict(
        zip(
            SIMPLE_TEXT_FEATURE_COLS,
            [
                "Words",
                "Sentences",
                "Words/sentence",
                "Exclamations",
                "Questions",
                "Caps (letters)",
            ],
        )
    )
    sub = sub.copy()
    sub["feat_lab"] = sub["feature"].map(label_map)
    order_lab = list(label_map.values())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(
        data=sub,
        x="feat_lab",
        y="mean",
        hue="propa_group",
        order=order_lab,
        ax=ax,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean (per post)")
    ax.set_title("Length & punctuation (NLTK sentences): mean by label group")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=16, ha="right")
    ax.legend(title="")
    fig.tight_layout()
    path = out_dir / "propa_simple_text_means_by_propa.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bw_fullcaps_mean_bars(summary_long: pd.DataFrame, out_dir: Path) -> Path | None:
    """Absolutist lexicon count vs all-caps word share."""
    sub = summary_long[
        summary_long["feature"].isin(BW_AND_FULLCAPS_COLS)
        & summary_long["propa_group"].isin(("propaganda", "not_propaganda"))
    ]
    if len(sub) < len(BW_AND_FULLCAPS_COLS) * 2:
        return None
    label_map = dict(
        zip(
            BW_AND_FULLCAPS_COLS,
            [
                "Absolutist words (count)",
                "All-caps words (share)",
            ],
        )
    )
    sub = sub.copy()
    sub["feat_lab"] = sub["feature"].map(label_map)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=sub,
        x="feat_lab",
        y="mean",
        hue="propa_group",
        order=list(label_map.values()),
        ax=ax,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean")
    ax.set_title("Absolutist wording & full caps word ratio")
    ax.legend(title="")
    fig.tight_layout()
    path = out_dir / "propa_bw_and_fullcaps_means_by_propa.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_emoji_mean_bars(summary_long: pd.DataFrame, out_dir: Path) -> Path | None:
    """Mean emoji clusters per post by label group."""
    sub = summary_long[
        summary_long["feature"].isin(EMOJI_FEATURE_COLS)
        & summary_long["propa_group"].isin(("propaganda", "not_propaganda"))
    ]
    if sub.empty or sub["propa_group"].nunique() < 2:
        return None
    sub = sub.copy()
    sub["feat_lab"] = "Emojis per post"
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.barplot(
        data=sub,
        x="feat_lab",
        y="mean",
        hue="propa_group",
        order=["Emojis per post"],
        ax=ax,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean")
    ax.set_title("Emoji count (Unicode clusters)")
    ax.legend(title="")
    fig.tight_layout()
    path = out_dir / "propa_emoji_means_by_propa.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_strip_or_violin(melted: pd.DataFrame, out_dir: Path) -> Path:
    """Small violin plots (shows density better than boxes for counts)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=melted,
        x="feature",
        y="value",
        hue="propa_group",
        order=FEATURE_COLS,
        ax=ax,
        split=True,
        inner="box",
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_title("Feature distributions (split violin: propaganda vs not propaganda)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    path = out_dir / "propa_features_violin_split.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args_input = LABELING_WITH_FEATURES_CSV
    args_output_table = PRELIMINARY_FEATURE_SUMMARY_CSV
    args_out_dir = FIGURES_DIR
    args_fdr_alpha = PRELIMINARY_FDR_ALPHA
    args_output_tests = PRELIMINARY_TESTS_CSV
    args_mi_random_state = PRELIMINARY_MI_RANDOM_STATE

    if not args_input.exists():
        raise SystemExit(f"Input not found: {args_input}")

    df = pd.read_csv(args_input, encoding="utf-8-sig")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing feature columns: {missing}")

    df = df.copy()
    df["propa_group"] = df["Propa"].map(propa_group_label)

    print("Row counts by Propa bucket")
    print(df["propa_group"].value_counts().sort_index().to_string())
    print()

    labeled = df[df["propa_group"].isin(("propaganda", "not_propaganda"))].copy()
    if len(labeled) < 10:
        raise SystemExit("Too few labeled propaganda / not-propaganda rows to summarize.")

    rows = []
    for name, grp in labeled.groupby("propa_group"):
        for col in FEATURE_COLS:
            s = pd.to_numeric(grp[col], errors="coerce").dropna()
            rows.append(
                {
                    "propa_group": name,
                    "feature": col,
                    "n": len(s),
                    "mean": s.mean(),
                    "std": s.std(),
                    "median": s.median(),
                }
            )

    summary_long = pd.DataFrame(rows)
    summary_long.to_csv(args_output_table, index=False)
    print(f"Wrote table: {args_output_table}")
    print()

    # Wide mean comparison (quick eyeball)
    pivot = summary_long.pivot(index="feature", columns="propa_group", values="mean")
    print("Mean by group (labeled rows only)")
    print(pivot.to_string())
    print()
    print("Cohen's d by feature (propaganda vs not_propaganda; pooled SD from labeled sample)")
    d_tab = _cohens_d_per_feature(summary_long).reindex(FEATURE_COLS).dropna()
    if len(d_tab):
        print(d_tab.to_string())
    else:
        print("(not computed)")
    print()

    mw_tests = mann_whitney_tests_vs_propa(labeled, fdr_alpha=args_fdr_alpha)
    if mw_tests is not None:
        mw_out = mw_tests.merge(
            _cohens_d_per_feature(summary_long).rename("cohens_d"),
            left_on="feature",
            right_index=True,
            how="left",
        )
        mi_pack = mutual_info_scores_vs_propa(
            labeled, FEATURE_COLS, random_state=args_mi_random_state
        )
        if mi_pack is not None:
            mi_series, n_mi, mi_discrete = mi_pack
            mw_out["mutual_info_classif"] = mw_out["feature"].map(mi_series)
            n_d = int(mi_discrete.sum())
            n_c = int((~mi_discrete).sum())
            print(
                f"mutual_info_classif: {n_mi} complete rows, "
                f"discrete_features=True for {n_d} cols / False for {n_c} (see "
                f"MUTUAL_INFO_CONTINUOUS_FEATURE_COLS in config), "
                f"random_state={args_mi_random_state}"
            )
        else:
            print(
                "(skipping mutual_info_classif: need sklearn, ≥10 complete rows, both classes, "
                "or install scikit-learn)"
            )
        cols = [
            "feature",
            "cohens_d",
            "mutual_info_classif",
            "p_mannwhitney",
            "p_fdr_bh",
            "sig_fdr",
            "statistic_mw",
            "n_propa",
            "n_not",
        ]
        mw_out = mw_out[[c for c in cols if c in mw_out.columns]]
        mw_out.to_csv(args_output_tests, index=False)
        print(
            "Mann-Whitney U vs not_propaganda (two-sided). "
            f"H0: identical distribution; H1: differs. FDR-BH at alpha = {args_fdr_alpha}"
        )
        print(f"Wrote tests: {args_output_tests}")
        with pd.option_context("display.max_rows", len(mw_out), "display.width", 120):
            print(mw_out.to_string(index=False))
        print()
        print("Sorted by raw p_mannwhitney (smallest p first):")
        print(
            mw_out.sort_values("p_mannwhitney", na_position="last")
            .to_string(index=False)
        )
        print()
    else:
        mw_out = None
        print(
            "(scipy not installed: install scipy for Mann-Whitney + FDR and Cohen's d plot stars)\n"
        )

    args_out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titlesize": 12})

    melted = labeled.melt(
        id_vars=["propa_group"],
        value_vars=FEATURE_COLS,
        var_name="feature",
        value_name="value",
    )
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")

    saved = []
    saved.append(plot_propa_label_counts(df, args_out_dir))
    saved.append(plot_feature_mean_bars(summary_long, args_out_dir))
    p_d = plot_cohens_d_bars(
        summary_long,
        args_out_dir,
        mw_tests=mw_out,
        fdr_alpha=args_fdr_alpha,
    )
    if p_d is not None:
        saved.append(p_d)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=melted,
        x="feature",
        y="value",
        hue="propa_group",
        order=FEATURE_COLS,
        ax=ax,
        fliersize=2,
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_title(
        "Feature distributions: propaganda vs not propaganda (labeled rows only)"
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    p_box = args_out_dir / "preliminary_features_by_propa.png"
    fig.savefig(p_box, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p_box)

    try:
        saved.append(plot_strip_or_violin(melted, args_out_dir))
    except ValueError:
        # split violin can fail if too many zeros in a category
        print("(skipped split violin plot: insufficient variation in some cells)")

    p = plot_subset_boxplot(
        melted,
        LINK_FEATURE_COLS,
        args_out_dir / "propa_link_features_boxplot.png",
        "Links: distributions by label (labeled rows)",
        figsize=(9, 5),
    )
    if p:
        saved.append(p)

    p = plot_link_mean_bars(summary_long, args_out_dir)
    if p:
        saved.append(p)

    p = plot_link_own_share_boxplot(labeled, args_out_dir)
    if p:
        saved.append(p)

    p = plot_subset_boxplot(
        melted,
        DEP_SYNTAX_FEATURE_COLS,
        args_out_dir / "propa_dep_syntax_boxplot.png",
        "Dependency syntax: distributions by label (labeled rows)",
        figsize=(12, 5.2),
    )
    if p:
        saved.append(p)

    p = plot_dep_syntax_mean_bars(summary_long, args_out_dir)
    if p:
        saved.append(p)

    p = plot_subset_boxplot(
        melted,
        SIMPLE_TEXT_FEATURE_COLS,
        args_out_dir / "propa_simple_text_boxplot.png",
        "Length & punctuation: distributions by label (labeled rows)",
        figsize=(12, 5.2),
    )
    if p:
        saved.append(p)

    p = plot_simple_text_mean_bars(summary_long, args_out_dir)
    if p:
        saved.append(p)

    p = plot_subset_boxplot(
        melted,
        BW_AND_FULLCAPS_COLS,
        args_out_dir / "propa_bw_and_fullcaps_boxplot.png",
        "Absolutist lexicon & all-caps word share: distributions by label",
        figsize=(7, 5),
    )
    if p:
        saved.append(p)

    p = plot_bw_fullcaps_mean_bars(summary_long, args_out_dir)
    if p:
        saved.append(p)

    p = plot_subset_boxplot(
        melted,
        EMOJI_FEATURE_COLS,
        args_out_dir / "propa_emoji_boxplot.png",
        "Emoji count: distributions by label (labeled rows)",
        figsize=(5.5, 4.5),
    )
    if p:
        saved.append(p)

    p = plot_emoji_mean_bars(summary_long, args_out_dir)
    if p:
        saved.append(p)

    print("Figures:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
