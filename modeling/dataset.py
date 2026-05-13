"""
Merge two labeling sheets (Uku + Simon) with a feature table.

The modeling target is always the **consensus** score: mean of the two ``Propa`` ratings
on [0, 4] (``y_propa`` / ``propa_consensus``). Individual coders are not exported.

Telegram ``id`` is only unique within a channel, so we join on ``(channel, telegram_id)``
with the feature table using ``(channel, id)``.

``channel`` is normalized (strip, lower, leading ``@`` removed) so two exports that
differ only by handle formatting still match.

For benchmark scripts, call :func:`validate_modeling_table` once after loading
``modeling_table.csv`` instead of scattering ``if "channel" in df`` checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_modeling_table(
    df: pd.DataFrame,
    *,
    need_text: bool = False,
    need_tyyp: bool = False,
) -> None:
    """
    Columns produced by :func:`build_modeling_table` (official ``export-table`` path).

    Call once after loading a modeling CSV so downstream code does not branch on
    ``if "channel" in df.columns`` etc.
    """
    required = ["channel", "telegram_id", "y_propa"]
    if need_text:
        required.append("text")
    if need_tyyp:
        required.append("y_tyyp")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Modeling table missing columns {missing}. "
            "Build with: python run_model_benchmark.py export-table"
        )


def normalize_channel(value: object) -> str:
    """Normalize a single Telegram channel handle (strip, lowercase, drop leading @)."""
    return str(value or "").strip().lower().lstrip("@")


def normalize_channel_column(df: pd.DataFrame, col: str = "channel") -> pd.DataFrame:
    """Return a copy with ``col`` normalized in place; raises if the column is missing."""
    if col not in df.columns:
        raise ValueError(f"DataFrame missing column {col!r}")
    out = df.copy()
    out[col] = out[col].astype(str).str.strip().str.lower().str.lstrip("@")
    return out


def add_normalized_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``channel`` from ``Channel`` or ``channel`` for stable merge keys."""
    if "channel" in df.columns:
        src = "channel"
    elif "Channel" in df.columns:
        src = "Channel"
    else:
        raise ValueError("CSV must contain Channel or channel")
    out = df.copy()
    out["channel"] = (
        out[src].astype(str).str.strip().str.lower().str.lstrip("@")
    )
    return out


def _tyyp_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.replace("ü", "u").lower()
        if cl == "tuup" or c == "Tüüp":
            return c
    return None


def load_labeling_pair(
    uku_path: Path | str,
    simon_path: Path | str,
    *,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """Inner-merge two labeling sheets on (channel, telegram_id)."""
    uku = pd.read_csv(uku_path, encoding=encoding)
    simon = pd.read_csv(simon_path, encoding=encoding)

    uku = add_normalized_channel(uku)
    simon = add_normalized_channel(simon)

    if "telegram_id" not in uku.columns or "telegram_id" not in simon.columns:
        raise ValueError("Labeling CSVs must contain telegram_id")

    uku = uku.rename(columns={"Propa": "propa_uku"})
    simon = simon.rename(columns={"Propa": "propa_simon"})

    for df in (uku, simon):
        df["_tid"] = pd.to_numeric(df["telegram_id"], errors="coerce").astype("Int64")

    ty_u = _tyyp_column(uku)
    ty_s = _tyyp_column(simon)
    if ty_u:
        uku = uku.rename(columns={ty_u: "tyyp_uku"})
    if ty_s:
        simon = simon.rename(columns={ty_s: "tyyp_simon"})

    simon_cols = ["channel", "_tid", "propa_simon"] + (
        ["tyyp_simon"] if "tyyp_simon" in simon.columns else []
    )
    merged = uku.merge(
        simon[simon_cols],
        on=["channel", "_tid"],
        how="inner",
        suffixes=("", "_s"),
    )
    merged["telegram_id"] = merged["_tid"]
    merged = merged.drop(columns=["_tid"], errors="ignore")

    print(
        f"  Inner merge on (channel, telegram_id): {len(merged)} rows "
        f"(Uku {len(uku)}, Simon {len(simon)})."
    )

    for c in ("propa_uku", "propa_simon"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged["propa_consensus"] = (
        (merged["propa_uku"] + merged["propa_simon"]) / 2.0
    ).clip(0.0, 4.0)

    text_col = "Text" if "Text" in merged.columns else "text"
    merged["text"] = merged[text_col].fillna("").astype(str)

    return merged


def merge_features(
    labeling: pd.DataFrame,
    features_path: Path | str,
    *,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """Left join features where (channel, telegram_id) matches (channel, id)."""
    feat = pd.read_csv(features_path, encoding=encoding, low_memory=False)
    if "id" not in feat.columns:
        raise ValueError(f"Features CSV must have 'id' (message id): {features_path}")

    lab = add_normalized_channel(labeling.copy())
    lab["_tid"] = pd.to_numeric(lab["telegram_id"], errors="coerce").astype("Int64")

    feat = add_normalized_channel(feat)
    feat["_id_join"] = pd.to_numeric(feat["id"], errors="coerce").astype("Int64")
    feat = feat.drop(
        columns=[c for c in ("text", "sample nr", "Propa", "Username") if c in feat.columns],
        errors="ignore",
    )

    merged = lab.merge(
        feat,
        left_on=["channel", "_tid"],
        right_on=["channel", "_id_join"],
        how="left",
        suffixes=("", "_feat"),
    )
    merged = merged.drop(
        columns=[c for c in ("_tid", "_id_join") if c in merged.columns],
        errors="ignore",
    )

    n_ok = int(merged["id"].notna().sum())
    n_lab = len(merged)
    print(
        f"Feature merge: {n_ok}/{n_lab} rows matched on (channel, telegram_id) == (channel, id)"
    )
    if n_ok < n_lab:
        if n_ok > 0 and n_lab == len(feat):
            print(
                "  Likely cause: labeling_with_features was built from a **different 500-post sample** "
                "than uku-labeling/simon-labeling (same row count, different message ids)."
            )
        print(
            "  Fix: run the feature pipeline from the **same** labeling export CSV the coders used, e.g.\n"
            "    python main.py --from-labeling-csv data/<that_export>.csv "
            "--start-from 03_language --export-csv labeling_with_features"
        )

    return merged


def build_modeling_table(
    uku_path: Path | str,
    simon_path: Path | str,
    features_path: Path | str | None,
    *,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """
    Build rows with consensus ``y_propa`` (= ``propa_consensus``, mean of both coders on [0, 4]).
    Drops per-coder ``Propa`` columns from the exported frame.
    """
    df = load_labeling_pair(uku_path, simon_path, encoding=encoding)
    if features_path:
        df = merge_features(df, features_path, encoding=encoding)

    df["y_propa"] = df["propa_consensus"]
    df = df.drop(columns=["propa_uku", "propa_simon"], errors="ignore")

    if "tyyp_uku" in df.columns:
        df["y_tyyp"] = df["tyyp_uku"].fillna("").astype(str).str.strip()
        df.loc[df["y_tyyp"] == "", "y_tyyp"] = np.nan
    else:
        df["y_tyyp"] = np.nan

    return df
