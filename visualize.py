"""
Exploratory visualizations for scraped Telegram data.

Default input is the labeling pool (``data/pool_pre_sample.csv``): posts after
max length, min length, and target-language filters — same as pre-sample
pipeline output. Pass ``--input data/latest_output.csv`` only if you want
figures for the final random sample (e.g. 500 rows).

Filter funnel stats (``<=1000`` chars → ``>=80`` strip → Estonian) run automatically
from ``checkpoints/01_raw_data.pkl`` when that file exists. Use ``--no-funnel`` to
skip (faster, figures only), or ``--funnel-input path.csv`` to override. Language
detection on the full raw table can take a while.

Default paths and time grouping: ``config.py`` (VISUALIZE_INPUT_CSV, FIGURES_DIR, VISUALIZE_TIME_FREQ).

Usage:
    python visualize.py
    python visualize.py --no-funnel
    python visualize.py --funnel-input data/telegram_unfiltered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

import features
from config import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    MAX_POST_CHARS,
    MIN_POST_CHARS,
    LABELING_TARGET_LANG,
    TOPICS,
    VISUALIZE_INPUT_CSV,
    VISUALIZE_TIME_FREQ,
)
from utils import checkpoint_exists, load_checkpoint


def _first_pipeline_checkpoint_name() -> str:
    from pipeline import Pipeline

    return Pipeline().stages[0][0]


def _prepare_funnel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw scrape DataFrame like :func:`load_data` (text, date, text_len, channel)."""
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out = out.dropna(subset=["date"])
    if "text" not in out.columns:
        raise ValueError("Funnel data must include a 'text' column")
    out["text"] = out["text"].fillna("").astype(str)
    if "channel" not in out.columns:
        out["channel"] = "unknown"
    else:
        out["channel"] = out["channel"].fillna("unknown").astype(str)
    out["text_len"] = out["text"].str.len()
    return out


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    df["text"] = df["text"].fillna("").astype(str)
    if "channel" not in df.columns:
        df["channel"] = "unknown"
    else:
        df["channel"] = df["channel"].fillna("unknown").astype(str)
    for col in ("views", "forwards"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["text_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len().fillna(0).astype(int)
    return df


def match_topic(text: str, keywords: list[str]) -> bool:
    if not text or not keywords:
        return False
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def run_filter_funnel(
    df: pd.DataFrame,
    *,
    language_verbose: bool = True,
) -> tuple[list[tuple[str, int, int]], pd.DataFrame]:
    """
    Apply the same length + language filters as ``pipeline._filter_text_bounds``
    and ``03_language`` / ``03b_target_language``.

    Returns:
        (steps, final_df) where each step is
        ``(label, n_remaining, n_removed_in_this_step)``.
    """
    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'text' column for the funnel")
    d = df.copy()
    d["text"] = d["text"].fillna("").astype(str)
    n_raw = len(d)
    d = features.filter_by_max_text_length(d, verbose=False)
    n1 = len(d)
    d = features.filter_by_min_text_length(d, verbose=False)
    n2 = len(d)
    d = features.add_language(d, verbose=language_verbose)
    d = features.filter_by_target_language(d, verbose=False)
    n3 = len(d)
    steps: list[tuple[str, int, int]] = [
        ("Scraped (input rows)", n_raw, 0),
        (f"After <={MAX_POST_CHARS} characters", n1, n_raw - n1),
        (f"After >={MIN_POST_CHARS} characters (strip length)", n2, n1 - n2),
        (
            f"After language == {LABELING_TARGET_LANG!r} (lingua)",
            n3,
            n2 - n3,
        ),
    ]
    return steps, d


def plot_filter_funnel(steps: list[tuple[str, int, int]], out_dir: Path) -> Path:
    """Bar chart: remaining posts after each filter stage."""
    labels = [s[0] for s in steps]
    counts = [s[1] for s in steps]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, counts, color=sns.color_palette("deep", n_colors=len(labels)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Posts remaining")
    ax.margins(x=0.02)
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = out_dir / "00_filter_funnel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_filter_funnel_table(steps: list[tuple[str, int, int]]) -> None:
    print("\n--- Filter funnel (length + language) ---")
    w = max(len(s[0]) for s in steps)
    for label, kept, removed in steps:
        extra = f"  (removed in step: {removed:,})" if removed else ""
        print(f"  {label:{w}}  {kept:>8,} rows{extra}")
    final_kept = steps[-1][1] if steps else 0
    print(f"  -> Final count for labeling / analysis: {final_kept:,} posts")


def add_topic_columns(df: pd.DataFrame, topics: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for name, kws in topics.items():
        col = f"topic_{name}"
        out[col] = out["text"].apply(lambda t: match_topic(t, kws))
    return out


def plot_channel_pie(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df["channel"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("Set2", n_colors=len(counts))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
        colors=colors,
        pctdistance=0.75,
        labeldistance=1.05,
    )
    for t in autotexts:
        t.set_fontsize(9)
    fig.tight_layout()
    path = out_dir / "01_channels_pie.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_temporal_stacked(
    df: pd.DataFrame,
    out_dir: Path,
    freq: str = "MS",
) -> Path:
    """
    Stacked bar: post counts per time bucket, stacked by source.
    freq: pandas offset alias, e.g. 'MS' month start, 'W' week.
    """
    d = df.set_index("date").sort_index()
    g = (
        d.groupby([pd.Grouper(freq=freq), "channel"])
        .size()
        .rename("posts")
        .reset_index()
    )
    pivot = g.pivot(index="date", columns="channel", values="posts").fillna(0)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(pivot)
    x = np.arange(n, dtype=float)
    bottom = np.zeros(n)
    cmap = plt.get_cmap("tab20")
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values.astype(float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=col,
            width=1.0,
            align="edge",
            color=cmap(i % 20),
        )
        bottom += vals

    if freq.upper().startswith("M"):
        tick_labels = [d.strftime("%Y-%m") for d in pivot.index]
    elif freq.upper().startswith("W"):
        tick_labels = [f"{d.strftime('%Y-%m-%d')}" for d in pivot.index]
    else:
        tick_labels = [d.strftime("%Y-%m-%d") for d in pivot.index]

    ax.set_xlim(0, n)
    ax.margins(x=0)
    ax.set_xticks(x + 0.5)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=8)
    ax.set_xlabel("Period")
    ax.set_ylabel("Number of posts")
    ax.legend(title="Channel", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(rect=[0.02, 0.14, 0.86, 0.98])
    path = out_dir / "02_temporal_stacked_by_channel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_posts_timeline_total(df: pd.DataFrame, out_dir: Path, freq: str = "MS") -> Path:
    """Line chart: total posts per period (all sources)."""
    s = df.set_index("date").sort_index().resample(freq).size()
    fig, ax = plt.subplots(figsize=(10, 4))
    s.plot(ax=ax, marker="o", markersize=4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Posts")
    ax.grid(True, alpha=0.3)
    if freq.upper().startswith("M"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.98])
    path = out_dir / "03_posts_timeline_total.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_text_length_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Histogram of character length.

    If every row is within the pipeline bounds (min strip length, max raw
    length, matching ``features``), we bin only on ``[MIN_POST_CHARS,
    MAX_POST_CHARS]`` using **raw** length (same as the max-length filter), lock
    the x axis to that range, and place a tick at the minimum so the first bin
    is not read as a separate “0–200” hump. ``ax.margins`` is not applied on x
    after the plot, because it can re-trigger autoscaling and offset the range.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    text = df["text"].fillna("").astype(str)
    tlen = text.str.len()
    tlen_strip = text.str.strip().str.len()
    # Same notion as the pipeline: min on strip, max on raw (features.py).
    use_pool_bounds = len(tlen) and (
        tlen_strip.min() >= MIN_POST_CHARS and tlen.max() <= MAX_POST_CHARS
    )
    if use_pool_bounds:
        lo, hi = int(MIN_POST_CHARS), int(MAX_POST_CHARS)
        n_bins = 40
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(tlen, bins=bin_edges)
        w = float(bin_edges[1] - bin_edges[0])
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        color = sns.color_palette("deep", n_colors=1)[0]
        ax.bar(
            centers,
            counts,
            width=w * 0.98,
            align="center",
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Characters")
        ax.set_ylabel("Count")
        ax.set_xlim(lo, hi)
        ax.set_xticks([80, 200, 400, 600, 800, 1000])
        ax.set_autoscalex_on(False)
        ax.margins(y=0.02)
    else:
        sns.histplot(
            tlen,
            bins=50,
            binrange=(0, 1000),
            ax=ax,
            kde=False,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Characters")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.margins(x=0, y=0.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "04_text_length_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_views_by_channel(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Box plot of view counts per channel (linear scale)."""
    if "views" not in df.columns:
        return None
    sub = df[df["views"] > 0].copy()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=sub, x="channel", y="views", ax=ax, showfliers=False)
    ax.set_ylim(bottom=0)
    ax.margins(y=0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    path = out_dir / "05_views_by_channel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_topic_counts(df: pd.DataFrame, topics: dict[str, list[str]], out_dir: Path) -> Path:
    if not topics:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No topics defined in config.TOPICS", ha="center", va="center")
        ax.axis("off")
        path = out_dir / "06_topics_bar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    tagged = add_topic_columns(df, topics)
    counts = {name: int(tagged[f"topic_{name}"].sum()) for name in topics}
    summary = pd.Series(counts).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 1.2), 5))
    summary.plot(kind="bar", ax=ax, color=sns.color_palette("husl", n_colors=len(summary)))
    ax.set_ylabel("Posts with ≥1 keyword")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path = out_dir / "06_topics_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_topic_cooccurrence_heatmap(df: pd.DataFrame, topics: dict[str, list[str]], out_dir: Path) -> Path:
    """Shows how often topic pairs co-occur in the same post."""
    names = list(topics.keys())
    if len(names) < 2:
        path = out_dir / "07_topic_cooccurrence.png"
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Add ≥2 topics for co-occurrence heatmap", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    tagged = add_topic_columns(df, topics)
    cols = [f"topic_{n}" for n in names]
    mat = tagged[cols].astype(int).T @ tagged[cols].astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=names, yticklabels=names)
    fig.tight_layout()
    path = out_dir / "07_topic_cooccurrence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Visualize Telegram scraped data")
    parser.add_argument(
        "--input",
        type=Path,
        default=VISUALIZE_INPUT_CSV,
        help="CSV for figures (default: config.VISUALIZE_INPUT_CSV)",
    )
    parser.add_argument(
        "--funnel-input",
        type=Path,
        default=None,
        help="Unfiltered CSV for the funnel (overrides the first pipeline checkpoint pkl if set)",
    )
    parser.add_argument(
        "--no-funnel",
        action="store_true",
        help="Do not run filter funnel (skip 01_raw_data.pkl / language detection even if present)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    out_dir = FIGURES_DIR
    time_freq = VISUALIZE_TIME_FREQ
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    df = load_data(args.input)
    print(f"Loaded {len(df)} rows from {args.input} (figures)")

    funnel_df: pd.DataFrame | None = None
    if args.funnel_input is not None:
        if not args.funnel_input.exists():
            raise SystemExit(f"--funnel-input not found: {args.funnel_input}")
        funnel_df = load_data(args.funnel_input)
        print(f"Funnel: using {len(funnel_df)} rows from {args.funnel_input}")
    elif not args.no_funnel:
        stage = _first_pipeline_checkpoint_name()
        if checkpoint_exists(stage):
            raw = load_checkpoint(stage)
            if not isinstance(raw, pd.DataFrame):
                print(f"Funnel: checkpoint {stage!r} is not a DataFrame — skip")
            else:
                try:
                    funnel_df = _prepare_funnel_dataframe(raw)
                except ValueError as e:
                    print(f"Funnel: {e} — skip")
                    funnel_df = None
                else:
                    p = CHECKPOINTS_DIR / f"{stage}.pkl"
                    print(f"Funnel: using {len(funnel_df):,} rows from checkpoint {stage} ({p})")
        else:
            print(
                f"Funnel: no checkpoint {stage!r} under checkpoints/ — skip "
                "(add the pkl, use --funnel-input, or pass --no-funnel to hide this message)"
            )

    paths: list[Path] = []
    if funnel_df is not None:
        steps, _ = run_filter_funnel(funnel_df, language_verbose=True)
        print_filter_funnel_table(steps)
        summary = pd.DataFrame(
            [{"step": a, "posts_remaining": b, "removed_in_step": c} for a, b, c in steps]
        )
        csv_path = out_dir / "filter_funnel_counts.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")
        paths.append(plot_filter_funnel(steps, out_dir))

    paths.append(plot_channel_pie(df, out_dir))
    paths.append(plot_temporal_stacked(df, out_dir, freq=time_freq))
    paths.append(plot_posts_timeline_total(df, out_dir, freq=time_freq))
    paths.append(plot_text_length_distribution(df, out_dir))
    p_views = plot_views_by_channel(df, out_dir)
    if p_views is not None:
        paths.append(p_views)
    paths.append(plot_topic_counts(df, TOPICS, out_dir))
    paths.append(plot_topic_cooccurrence_heatmap(df, TOPICS, out_dir))

    print("Saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
