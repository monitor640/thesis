"""
Build ``data/holdout_llm_predictions.csv`` and ``holdout_llm_with_features_predictions.csv``
from LLM JSONL checkpoints + the fixed holdout manifest (same rows as sklearn/BERT).

Columns match the BERT holdout file shape: ``channel``, ``telegram_id``, ``y_true``, and
one prediction column (``pred_llm`` / ``pred_llm_with_features``).

  python -m modeling.holdout_llm_csvs [--model MODEL_ID]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    HOLDOUT_LLM_PREDS_CSV,
    HOLDOUT_LLM_WITH_FEATURES_PREDS_CSV,
    HOLDOUT_MANIFEST_CSV,
    MODELING_TABLE_CSV,
    llm_checkpoint_jsonl,
)
from modeling.dataset import normalize_channel_column, validate_modeling_table

_VARIANT_EXPORT: dict[str, tuple[Path, str]] = {
    "no_features": (HOLDOUT_LLM_PREDS_CSV, "pred_llm"),
    "with_features": (HOLDOUT_LLM_WITH_FEATURES_PREDS_CSV, "pred_llm_with_features"),
}


def read_llm_jsonl_pred_df(jsonl_path: Path) -> pd.DataFrame:
    """One row per (channel, telegram_id); last JSONL line wins; skips errors / null pred."""
    if not jsonl_path.is_file():
        return pd.DataFrame(columns=["channel", "telegram_id", "pred"])
    rows: list[dict[str, object]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if j.get("error"):
            continue
        if j.get("pred") is None:
            continue
        tid = j.get("telegram_id")
        if tid is None:
            continue
        ch = j.get("channel")
        rows.append(
            {
                "channel": "" if ch is None else str(ch),
                "telegram_id": int(tid),
                "pred": float(j["pred"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["channel", "telegram_id", "pred"])
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["channel", "telegram_id"], keep="last")


def _truth_and_merge(df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Same join rules as ``run_model_benchmark._llm_truth_and_merge``."""
    truth = normalize_channel_column(
        df[["channel", "telegram_id", "y_propa"]].rename(columns={"y_propa": "y_true"})
    )
    preds = pred_df.dropna(subset=["pred"]).copy()
    preds["pred"] = pd.to_numeric(preds["pred"], errors="coerce")
    preds = preds.dropna(subset=["pred"])
    if "channel" in preds.columns:
        preds = normalize_channel_column(preds)
        m = truth.merge(preds, on=["channel", "telegram_id"], how="inner")
    else:
        m = truth.merge(preds, on="telegram_id", how="inner")
    m["y_true"] = pd.to_numeric(m["y_true"], errors="coerce")
    return m.dropna(subset=["y_true", "pred"])


def _finalize_holdout_frame(sub: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = sub[["channel", "telegram_id", "y_true", "pred"]].copy()
    out = out.rename(columns={"pred": pred_col})
    out["telegram_id"] = pd.to_numeric(out["telegram_id"], errors="coerce").astype("Int64")
    if out["telegram_id"].isna().any():
        raise ValueError("holdout LLM export: invalid telegram_id")
    out["telegram_id"] = out["telegram_id"].astype(int)
    return out


def write_holdout_llm_csv_from_merged(m: pd.DataFrame, variant: str) -> None:
    """Called from ``cmd_llm`` after a successful full-table merge."""
    if variant not in _VARIANT_EXPORT:
        return
    if not HOLDOUT_MANIFEST_CSV.is_file():
        return
    out_path, pred_col = _VARIANT_EXPORT[variant]
    ho = normalize_channel_column(
        pd.read_csv(HOLDOUT_MANIFEST_CSV, encoding="utf-8-sig")
    )
    sub = ho.merge(m, on=["channel", "telegram_id"], how="inner")
    if len(sub) == 0:
        print(f"[LLM holdout CSV] skip {out_path.name}: 0 rows matched manifest.", flush=True)
        return
    if len(sub) != len(ho):
        print(
            f"[LLM holdout CSV] warning: matched {len(sub)} rows, manifest has {len(ho)}.",
            flush=True,
        )
    out = _finalize_holdout_frame(sub, pred_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote holdout LLM predictions: {out_path}", flush=True)


def write_holdout_llm_csv_for_variant(
    *,
    variant: str,
    model_id: str | None = None,
    modeling_table: Path | None = None,
    manifest: Path | None = None,
) -> bool:
    """Rebuild one CSV from disk (JSONL + modeling table + manifest). Returns True if written."""
    if variant not in _VARIANT_EXPORT:
        raise ValueError(f"unknown variant: {variant!r}")
    out_path, pred_col = _VARIANT_EXPORT[variant]
    ck = llm_checkpoint_jsonl(variant, model_id=model_id)
    if not ck.is_file():
        print(f"[LLM holdout CSV] skip {out_path.name}: missing {ck}", flush=True)
        return False
    table = modeling_table or MODELING_TABLE_CSV
    man = manifest or HOLDOUT_MANIFEST_CSV
    if not man.is_file():
        print(f"[LLM holdout CSV] skip {out_path.name}: missing {man}", flush=True)
        return False
    df = pd.read_csv(table, encoding="utf-8-sig")
    validate_modeling_table(df, need_text=True)
    pred_df = read_llm_jsonl_pred_df(ck)
    m = _truth_and_merge(df, pred_df)
    ho = normalize_channel_column(pd.read_csv(man, encoding="utf-8-sig"))
    sub = ho.merge(m, on=["channel", "telegram_id"], how="inner")
    if len(sub) == 0:
        print(f"[LLM holdout CSV] skip {out_path.name}: 0 manifest matches.", flush=True)
        return False
    if len(sub) != len(ho):
        print(
            f"[LLM holdout CSV] warning: matched {len(sub)} rows, manifest has {len(ho)}.",
            flush=True,
        )
    out = _finalize_holdout_frame(sub, pred_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}", flush=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Export holdout LLM prediction CSVs from JSONL.")
    p.add_argument(
        "--model",
        default=None,
        help="OpenAI model id (default: config.LLM_MODEL / env).",
    )
    args = p.parse_args()
    for variant in ("no_features", "with_features"):
        write_holdout_llm_csv_for_variant(variant=variant, model_id=args.model)


if __name__ == "__main__":
    main()
