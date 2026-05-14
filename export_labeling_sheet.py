"""
Export sampled rows to a CSV that matches the manual labeling Google Sheet layout.

Columns (same order as your sheet):
  sample nr | Text | Username | Channel | Propa | Sildistaja

Username uses 'username' (@handle from Telegram) when present, else 'post_author'
(channel signature), else other legacy sender columns if any.

By default, only rows that pass the same rules as the pipeline labeling pool are
exported: max/min length (config) and target language (Estonian by default).

Default pool/output paths: ``config.py`` (POOL_PRE_SAMPLE_CSV, LABELING_EXPORT_CSV).

Usage:
  python export_labeling_sheet.py
  python export_labeling_sheet.py --sample 500
  python export_labeling_sheet.py --sample 500 --exclude-text-from data/prior_export.csv --with-meta

Refresh (replace rows that fail filters; labeled rows kept by default):
  python export_labeling_sheet.py --refresh data/labeling_export.csv

Google Sheets:
  1. File > Import > Upload > select the CSV > "Insert new sheet" or replace a tab.
  2. OR: keep your template with dropdowns on Propa — import CSV into a NEW tab,
     then copy columns A–D (sample nr, Text, Username, Channel) from that tab into
     your template tab (paste values only), so dropdown columns stay intact.
  3. UTF-8: use utf-8-sig encoding (default here) so Estonian characters open correctly.

Optional: programmatic upload with Google Cloud + gspread (not included here).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    LABELING_EXPORT_CSV,
    LABELING_TARGET_LANG,
    MIN_POST_CHARS,
    POOL_PRE_SAMPLE_CSV,
    SAMPLE_RANDOM_STATE,
)
import features
from utils import load_checkpoint

# Exact headers for your annotation sheet
LABEL_COLUMNS = [
    "sample nr",
    "Text",
    "Username",
    "Channel",
    "Propa",
    "Sildistaja",
]

# Columns that indicate the annotator filled something in (preserve on refresh).
LABEL_ANNOTATION_COLS = ["Propa", "Sildistaja"]

META_COLUMNS = ("telegram_id", "post_date")


def _normalize_text_key(s: str) -> str:
    """Stable key for matching post text across pool and labeling exports."""
    t = str(s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return t


def _text_key_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_text_key)


def load_excluded_text_keys(paths: list[Path]) -> set[str]:
    """Unique non-empty normalized texts from prior labeling export(s) (column ``Text``)."""
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        d = pd.read_csv(path, encoding="utf-8-sig")
        if "Text" not in d.columns:
            raise ValueError(f"Exclusion CSV must have a 'Text' column: {path}")
        for k in _text_key_series(d["Text"]).tolist():
            if k:
                keys.add(k)
    return keys


def _load_pool_dataframe(pool_path: Path) -> pd.DataFrame:
    if pool_path.exists():
        return pd.read_csv(pool_path)
    df = load_checkpoint("04_pos_tags")
    if df is not None:
        print(
            f"Pool file not found ({pool_path}); loaded checkpoint 04_pos_tags instead."
        )
        return df
    raise FileNotFoundError(
        f"No pool CSV at {pool_path} and no checkpoint 04_pos_tags. "
        "Run the pipeline through stage 04 (or full run) first."
    )


def pick_username_column(df: pd.DataFrame) -> str | None:
    for c in ("username", "sender", "sender_username", "from_username", "user"):
        if c in df.columns:
            return c
    return None


def sheet_username_series(df: pd.DataFrame) -> pd.Series:
    """Prefer Telegram username, then channel post_author, then legacy columns."""
    n = len(df)
    idx = df.index

    if "username" in df.columns:
        u = df["username"].fillna("").astype(str).str.strip()
    else:
        u = pd.Series([""] * n, index=idx, dtype=str)

    if "post_author" in df.columns:
        pa = df["post_author"].fillna("").astype(str).str.strip()
    else:
        pa = pd.Series([""] * n, index=idx, dtype=str)

    out = u.where(u != "", pa)

    leg_col = pick_username_column(df)
    if leg_col and leg_col != "username":
        leg = df[leg_col].fillna("").astype(str).str.strip()
        out = out.where(out != "", leg)

    return out


def build_labeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    username_series = sheet_username_series(df)

    out = pd.DataFrame(
        {
            "sample nr": range(1, n + 1),
            "Text": df["text"].fillna("").astype(str),
            "Username": username_series.values,
            "Channel": (
                df["channel"].fillna("").astype(str).values
                if "channel" in df.columns
                else pd.Series([""] * n, dtype=str).values
            ),
            "Propa": [""] * n,
            "Sildistaja": [""] * n,
        }
    )
    # Ensure column order
    return out[LABEL_COLUMNS]


def _row_is_labeled(row: pd.Series) -> bool:
    return any(str(row.get(c, "") or "").strip() for c in LABEL_ANNOTATION_COLS)


def _apply_labeling_filters(
    df: pd.DataFrame,
    min_chars: int | None = None,
    target_lang: str | None = None,
) -> pd.DataFrame:
    """Same length + language rules as the pipeline labeling pool."""
    df = features.filter_by_max_text_length(df)
    df = features.filter_by_min_text_length(df, min_chars=min_chars)
    if "language" not in df.columns:
        df = features.add_language(df)
    df = features.filter_by_target_language(df, lang=target_lang)
    return df


def _text_passes_criteria(
    text: str,
    min_chars: int,
    target_lang: str,
) -> bool:
    s = str(text or "").strip()
    if len(s) < min_chars:
        return False
    return features.detect_post_language(s) == target_lang.lower()


def _gather_used_pool_ids(existing: pd.DataFrame, pool: pd.DataFrame) -> set:
    """IDs from the pool already represented by rows we keep (for sampling without duplicates)."""
    used: set = set()
    if "id" not in pool.columns:
        return used
    text_col = pool["text"].fillna("").astype(str).str.strip()
    for _, row in existing.iterrows():
        tid = row.get("telegram_id")
        if tid is not None and str(tid).strip() != "" and not (
            isinstance(tid, float) and pd.isna(tid)
        ):
            try:
                used.add(int(float(tid)))
            except (TypeError, ValueError):
                pass
            continue
        rt = str(row.get("Text", "")).strip()
        if not rt:
            continue
        m = pool.loc[text_col == rt]
        if len(m):
            used.update(int(x) for x in m["id"].tolist())
    return used


def refresh_labeling_sheet(
    existing: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    random_state: int,
    min_chars: int | None = None,
    target_lang: str | None = None,
    preserve_labeled: bool = True,
) -> pd.DataFrame:
    """
    Replace rows that fail length/language criteria with new draws from pool.
    Rows with any annotation filled in are kept by default even if they violate.
    """
    mc = min_chars if min_chars is not None else MIN_POST_CHARS
    lang = (target_lang or LABELING_TARGET_LANG).lower()

    pool = _apply_labeling_filters(pool, min_chars=mc, target_lang=lang)
    if pool.empty:
        raise ValueError("Pool is empty after applying labeling filters.")

    if "id" not in pool.columns:
        raise ValueError("Pool must include an 'id' column for refresh (pipeline CSV).")

    labeled_violations: list[int] = []

    replace_flags: list[bool] = []
    for i, row in existing.iterrows():
        ok = _text_passes_criteria(str(row.get("Text", "")), mc, lang)
        labeled = _row_is_labeled(row)
        if ok:
            replace_flags.append(False)
        elif labeled and preserve_labeled:
            labeled_violations.append(int(row.get("sample nr", i + 1)))
            replace_flags.append(False)
        else:
            replace_flags.append(True)

    if labeled_violations:
        print(
            "Warning: kept labeled rows that still violate filters (sample nr): "
            f"{labeled_violations[:20]}{'...' if len(labeled_violations) > 20 else ''}"
        )
        print(
            "  Re-export with --no-preserve-labeled to replace them (drops those labels)."
        )

    n_replace = sum(replace_flags)
    if n_replace == 0:
        out = existing.copy()
        out["sample nr"] = range(1, len(out) + 1)
        base_cols = [c for c in out.columns if c in LABEL_COLUMNS or c in META_COLUMNS]
        return out[base_cols]

    keep_indices = [j for j, rep in enumerate(replace_flags) if not rep]
    scratch = existing.iloc[keep_indices]
    used = _gather_used_pool_ids(scratch, pool)
    eligible = pool[~pool["id"].isin(used)].copy()
    if len(eligible) < n_replace:
        raise ValueError(
            f"Not enough pool rows: need {n_replace} replacements, "
            f"only {len(eligible)} eligible after excluding already-used ids."
        )

    new_draw = eligible.sample(n=n_replace, random_state=random_state).reset_index(
        drop=True
    )

    meta_in_sheet = [c for c in META_COLUMNS if c in existing.columns]
    out_rows: list[dict] = []
    new_i = 0
    for j in range(len(existing)):
        r = existing.iloc[j]
        if not replace_flags[j]:
            row_dict = {c: r.get(c, "") for c in LABEL_COLUMNS}
            for mc in meta_in_sheet:
                row_dict[mc] = r.get(mc, "")
            out_rows.append(row_dict)
        else:
            sub = new_draw.iloc[[new_i]]
            new_i += 1
            lab = build_labeling_frame(sub)
            row_dict = {c: lab[c].iloc[0] for c in LABEL_COLUMNS}
            if "telegram_id" in meta_in_sheet and "id" in sub.columns:
                row_dict["telegram_id"] = sub["id"].iloc[0]
            if "post_date" in meta_in_sheet and "date" in sub.columns:
                row_dict["post_date"] = sub["date"].iloc[0]
            out_rows.append(row_dict)

    out = pd.DataFrame(out_rows)
    out["sample nr"] = range(1, len(out) + 1)
    front = ["sample nr"] + [c for c in LABEL_COLUMNS if c != "sample nr"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]
    print(f"Refresh: replaced {n_replace} row(s); {len(out)} total rows.")
    return out


def main():
    p = argparse.ArgumentParser(description="Export CSV for Google Sheets labeling")
    p.add_argument(
        "--input",
        type=Path,
        default=POOL_PRE_SAMPLE_CSV,
        help="Source CSV (default: config.POOL_PRE_SAMPLE_CSV)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=LABELING_EXPORT_CSV,
        help="Output CSV path (default: config.LABELING_EXPORT_CSV)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N rows after filters (optional)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=SAMPLE_RANDOM_STATE,
        help="Random seed for --sample / --refresh",
    )
    p.add_argument(
        "--with-meta",
        action="store_true",
        help="Append telegram_id and post_date columns after the 8 labeling columns",
    )
    p.add_argument(
        "--refresh",
        type=Path,
        default=None,
        metavar="CSV",
        help="Existing labeling CSV: swap out rows that fail filters, keep labels elsewhere",
    )
    p.add_argument(
        "--pool",
        type=Path,
        default=POOL_PRE_SAMPLE_CSV,
        help="Pool CSV for --refresh (default: config.POOL_PRE_SAMPLE_CSV)",
    )
    p.add_argument(
        "--no-preserve-labeled",
        action="store_true",
        help="With --refresh, also replace violating rows that already have labels",
    )
    p.add_argument(
        "--exclude-text-from",
        type=Path,
        action="append",
        default=[],
        metavar="CSV",
        help="Prior labeling export(s): drop pool rows whose text matches any row's Text "
        "(normalized). Use multiple times for several files. Does not use sample nr / id.",
    )
    args = p.parse_args()

    if args.refresh:
        if not args.refresh.exists():
            raise SystemExit(f"--refresh file not found: {args.refresh}")
        existing = pd.read_csv(args.refresh, encoding="utf-8-sig")
        try:
            pool = _load_pool_dataframe(args.pool)
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e
        label_df = refresh_labeling_sheet(
            existing,
            pool,
            random_state=args.random_state,
            min_chars=None,
            target_lang=None,
            preserve_labeled=not args.no_preserve_labeled,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        label_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Wrote: {args.output}")
        print(f"Columns: {list(label_df.columns)}")
        return

    if not args.input.exists():
        raise SystemExit(
            f"Input not found: {args.input}\n"
            "Use data/pool_pre_sample.csv after pipeline run, or pass --input path."
        )

    df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise SystemExit("Input CSV must contain a 'text' column")

    df = _apply_labeling_filters(df, min_chars=None, target_lang=None)
    print(f"After labeling filters: {len(df)} rows")
    if df.empty:
        raise SystemExit("No rows left after filters; widen criteria or check input.")

    if args.exclude_text_from:
        ex_paths = args.exclude_text_from
        ex_keys = load_excluded_text_keys(ex_paths)
        tk = _text_key_series(df["text"])
        before = len(df)
        df = df.loc[~tk.isin(ex_keys)].copy().reset_index(drop=True)
        n_drop = before - len(df)
        print(
            f"Excluded {n_drop} pool row(s) matching {len(ex_keys)} unique prior text(s) "
            f"from {len(ex_paths)} file(s)"
        )
        if df.empty:
            raise SystemExit(
                "No rows left after --exclude-text-from; check exports match pool encoding/text."
            )

    if args.sample is not None and args.sample > 0:
        take = min(args.sample, len(df))
        if take < args.sample:
            print(
                f"Warning: only {len(df)} rows available after filters/exclusions; "
                f"sampling {take} (requested {args.sample})."
            )
        df = df.sample(n=take, random_state=args.random_state).reset_index(drop=True)
        print(f"Sampled {len(df)} rows (random_state={args.random_state})")
    else:
        print(f"Using all {len(df)} rows")

    label_df = build_labeling_frame(df)

    if args.with_meta:
        if "id" in df.columns:
            label_df["telegram_id"] = df["id"].values
        if "date" in df.columns:
            label_df["post_date"] = df["date"].values

    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote: {args.output}")
    print(f"Columns: {list(label_df.columns)}")


if __name__ == "__main__":
    main()
