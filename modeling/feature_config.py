"""
Tabular feature columns for sklearn / LLM-with-features: from config.FEATURE_COLS,
filtered by preliminary_propa_group_tests.csv using the thesis OR-rule (Cohen's d,
BH-adjusted p, or MI with side conditions). No JSON overrides.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from config import (
    FEATURE_COLS,
    FEATURE_THESIS_COHEN_D_ABS_MIN,
    FEATURE_THESIS_FDR_ALPHA,
    FEATURE_THESIS_MI_GATE_COHEN_D_ABS,
    FEATURE_THESIS_MI_GATE_P_UNC,
    FEATURE_THESIS_MI_MIN,
    PRELIMINARY_TESTS_CSV,
)

EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset(
    {
        "pos_tags",
        "text",
        "Text",
        "channel",
        "username",
        "Propa",
        "Sildistaja",
        "sample nr",
        "telegram_id",
        "post_date",
        "id",
        "language",
    }
)


def feature_selection_rule() -> str:
    """Human-readable summary of the active selection rule (for manifests)."""
    return (
        "thesis OR: "
        f"|d|>={FEATURE_THESIS_COHEN_D_ABS_MIN} "
        f"or q(BH)<{FEATURE_THESIS_FDR_ALPHA} "
        f"or (MI>={FEATURE_THESIS_MI_MIN} and (|d|>={FEATURE_THESIS_MI_GATE_COHEN_D_ABS} "
        f"or p_unc<{FEATURE_THESIS_MI_GATE_P_UNC}))"
    )


def _safe_float(x: object, default: float | None = None) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return v


def passes_thesis_feature_tests_row(row: pd.Series) -> bool:
    """True if this preliminary-tests row satisfies at least one thesis inclusion criterion."""
    d_abs = abs(_safe_float(row.get("cohens_d"), 0.0) or 0.0)
    p_unc = _safe_float(row.get("p_mannwhitney"), 1.0)
    p_bh = _safe_float(row.get("p_fdr_bh"))
    mi = _safe_float(row.get("mutual_info_classif"))

    if d_abs >= FEATURE_THESIS_COHEN_D_ABS_MIN:
        return True
    if p_bh is not None and p_bh < FEATURE_THESIS_FDR_ALPHA:
        return True
    if mi is not None and mi >= FEATURE_THESIS_MI_MIN:
        if d_abs >= FEATURE_THESIS_MI_GATE_COHEN_D_ABS or p_unc < FEATURE_THESIS_MI_GATE_P_UNC:
            return True
    return False


def load_modeling_features_from_config(
    tests_csv: Path | None = None,
) -> list[str]:
    """
    Intersect config.FEATURE_COLS with rows in preliminary tests CSV that pass
    :func:`passes_thesis_feature_tests_row`. If nothing passes, falls back to all
    FEATURE_COLS (with a warning) so pipelines still run.
    """
    path = Path(tests_csv) if tests_csv is not None else PRELIMINARY_TESTS_CSV
    if not path.exists():
        print(
            f"  [warn] {path} missing — run preliminary_propa_features.py first. "
            "Using full config.FEATURE_COLS for modeling."
        )
        return list(FEATURE_COLS)

    df = pd.read_csv(path)
    pool = set(FEATURE_COLS)
    selected: list[str] = []
    for _, row in df.iterrows():
        name = str(row["feature"]).strip()
        if name not in pool:
            continue
        if passes_thesis_feature_tests_row(row):
            selected.append(name)

    # Preserve order of FEATURE_COLS
    ordered = [c for c in FEATURE_COLS if c in selected]
    if not ordered:
        print(
            "  [warn] No features passed thesis filter — using full config.FEATURE_COLS."
        )
        return list(FEATURE_COLS)
    print(f"  Feature selection: {len(ordered)} columns ({feature_selection_rule()})")
    return ordered


def resolve_feature_columns(df: pd.DataFrame, names: list[str]) -> list[str]:
    resolved: list[str] = []
    for n in names:
        if n in EXCLUDE_FROM_FEATURES:
            continue
        if n not in df.columns:
            print(f"  [warn] feature column missing, skip: {n!r}")
            continue
        resolved.append(n)
    return resolved
