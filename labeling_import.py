"""
Turn a labeling export CSV (export_labeling_sheet layout) into a DataFrame
the pipeline can consume from stage 03_language onward (requires ``text``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_labeling_csv_for_pipeline(path: Path | str) -> pd.DataFrame:
    """
    Read utf-8-sig CSV with columns like Text, Channel, Username, optional telegram_id.

    Maps to pipeline names: text, channel, username, id.
    Keeps labeling columns (sample nr, Propa, …) for merging results back.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    raw = pd.read_csv(p, encoding="utf-8-sig")
    if "Text" not in raw.columns:
        raise ValueError(f"CSV must contain a 'Text' column: {p}")

    n = len(raw)
    out = pd.DataFrame(
        {
            "text": raw["Text"].fillna("").astype(str),
            "channel": (
                raw["Channel"].fillna("").astype(str)
                if "Channel" in raw.columns
                else pd.Series([""] * n, dtype=str)
            ),
            "username": (
                raw["Username"].fillna("").astype(str)
                if "Username" in raw.columns
                else pd.Series([""] * n, dtype=str)
            ),
        }
    )
    if "telegram_id" in raw.columns:
        out["id"] = raw["telegram_id"]
    elif "sample nr" in raw.columns:
        out["id"] = raw["sample nr"].apply(
            lambda x: int(float(x)) if pd.notna(x) and str(x).strip() != "" else None
        )
    else:
        out["id"] = range(len(out))

    if "post_date" in raw.columns:
        out["date"] = raw["post_date"]

    for c in ("sample nr", "Propa", "Sildistaja", "Tüüp", "Muu tüüp"):
        if c in raw.columns:
            out[c] = raw[c]

    return out
