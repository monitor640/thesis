"""Backfill RMSE fields into ``metrics_repeated_splits.json`` files.

For every saved seed row, adds ``rmse = sqrt(mse)``. For every summary block,
adds ``rmse_mean`` (mean of per-seed RMSE) and ``rmse_ci_low`` / ``rmse_ci_high``
(2.5 / 97.5 percentiles of per-seed RMSE) so RMSE is treated exactly like MSE,
MAE and R² in the existing schema.

Handles both shapes produced by the repeated-splits subcommands:

* **sklearn** (``checkpoints/modeling/metrics_repeated_splits.json``)
  – ``per_seed_rows`` is ``{model_name: [row, ...]}`` and ``summary`` is
  ``{model_name: {...}}``.
* **BERT** (``checkpoints/bert_propa/metrics_repeated_splits.json``)
  – ``per_seed_rows`` is a flat ``[row, ...]`` and ``summary`` is a single
  ``{...}`` block.

Usage examples::

    # In-place backfill of one JSON
    python scripts/add_rmse_to_repeated_splits.py \\
        checkpoints/modeling/metrics_repeated_splits.json

    # Read renamed BERT backup, write to the canonical name (no overwrite of backup)
    python scripts/add_rmse_to_repeated_splits.py \\
        --out checkpoints/bert_propa/metrics_repeated_splits.json \\
        checkpoints/bert_propa/metrics_repeated_splits_no_rmse.json

    # Dry-run: print what *would* change without writing
    python scripts/add_rmse_to_repeated_splits.py --dry-run \\
        checkpoints/bert_propa/metrics_repeated_splits_no_rmse.json

Note: ``rmse_mean`` is the **mean of per-seed RMSEs**, NOT ``sqrt(mse_mean)``.
By Jensen's inequality these are not equal in general; reporting the
per-seed-mean keeps RMSE on the same footing as MSE/MAE/R² (each summary stat
is computed from the same 500 per-seed numbers). If you also want the
``sqrt(mean_mse)`` "global" view, pass ``--also-sqrt-grand-mean``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _row_rmse(mse: Any) -> float:
    if mse is None:
        return float("nan")
    try:
        m = float(mse)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(m) or m < 0:
        return float("nan")
    return math.sqrt(m)


def _add_rmse_to_rows(rows: list[dict]) -> int:
    """Insert ``rmse`` into every row (in place). Returns the count updated."""
    updated = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        row["rmse"] = _row_rmse(row.get("mse"))
        updated += 1
    return updated


def _summary_rmse_block(rows: list[dict], also_sqrt_grand_mean: bool) -> dict:
    """RMSE summary stats computed from per-seed RMSE values."""
    rmses = np.array(
        [r.get("rmse", float("nan")) for r in rows if isinstance(r, dict)],
        dtype=float,
    )
    mses = np.array(
        [r.get("mse", float("nan")) for r in rows if isinstance(r, dict)],
        dtype=float,
    )
    extra: dict[str, float] = {}
    if also_sqrt_grand_mean:
        mse_mean = float(np.nanmean(mses)) if mses.size else float("nan")
        extra["rmse_sqrt_of_mean_mse"] = (
            float(math.sqrt(mse_mean))
            if math.isfinite(mse_mean) and mse_mean >= 0
            else float("nan")
        )
    if rmses.size == 0:
        return {"rmse_mean": float("nan"),
                "rmse_ci_low": float("nan"),
                "rmse_ci_high": float("nan"),
                **extra}
    return {
        "rmse_mean": float(np.nanmean(rmses)),
        "rmse_ci_low": float(np.nanquantile(rmses, 0.025)),
        "rmse_ci_high": float(np.nanquantile(rmses, 0.975)),
        **extra,
    }


def _backfill_one(
    data: dict,
    *,
    also_sqrt_grand_mean: bool,
) -> tuple[int, list[str]]:
    """Mutate ``data`` in place to add RMSE everywhere. Returns (n_rows_updated, model_labels)."""
    per_seed = data.get("per_seed_rows")
    summary = data.get("summary")
    n_rows_total = 0
    labels: list[str] = []

    if isinstance(per_seed, dict):
        if not isinstance(summary, dict):
            raise ValueError(
                "Inconsistent JSON: per_seed_rows is a dict but summary is not. "
                "Cannot backfill RMSE safely."
            )
        for name, rows in per_seed.items():
            if not isinstance(rows, list):
                continue
            n_rows_total += _add_rmse_to_rows(rows)
            block = summary.get(name)
            if not isinstance(block, dict):
                block = {}
                summary[name] = block
            block.update(_summary_rmse_block(rows, also_sqrt_grand_mean))
            labels.append(name)
    elif isinstance(per_seed, list):
        n_rows_total += _add_rmse_to_rows(per_seed)
        if not isinstance(summary, dict):
            summary = {}
            data["summary"] = summary
        summary.update(_summary_rmse_block(per_seed, also_sqrt_grand_mean))
        # use model_id when available so dry-run output reads cleanly
        labels.append(str(data.get("model_id", "(flat)")))
    else:
        raise ValueError(
            "Unrecognized per_seed_rows shape: expected dict-of-lists or flat list."
        )

    return n_rows_total, labels


def _format_block(label: str, block: dict) -> str:
    keys = ("mse_mean", "rmse_mean", "rmse_ci_low", "rmse_ci_high")
    parts = []
    for k in keys:
        v = block.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}=nan")
    if "rmse_sqrt_of_mean_mse" in block:
        v = block["rmse_sqrt_of_mean_mse"]
        if isinstance(v, (int, float)) and math.isfinite(v):
            parts.append(f"rmse_sqrt_of_mean_mse={v:.4f}")
    return f"  {label:<42s}  " + "  ".join(parts)


def process_file(
    src: Path,
    dest: Path,
    *,
    also_sqrt_grand_mean: bool,
    dry_run: bool,
) -> None:
    if not src.is_file():
        raise SystemExit(f"Input JSON not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))

    pre_summary = data.get("summary")
    pre_had_rmse = False
    if isinstance(pre_summary, dict):
        # detect both shapes
        first = next(iter(pre_summary.values()), None)
        if isinstance(first, dict) and "rmse_mean" in first:
            pre_had_rmse = True
        elif "rmse_mean" in pre_summary:
            pre_had_rmse = True

    n_rows, labels = _backfill_one(data, also_sqrt_grand_mean=also_sqrt_grand_mean)

    print(f"\n== {src} ==", flush=True)
    if pre_had_rmse:
        print(
            "  note: file already had rmse_* in summary — overwriting with freshly computed values.",
            flush=True,
        )
    print(f"  models: {len(labels)}   rows touched: {n_rows}", flush=True)
    summary = data.get("summary") or {}
    if isinstance(summary, dict) and isinstance(data.get("per_seed_rows"), dict):
        for name in labels:
            block = summary.get(name) or {}
            print(_format_block(name, block), flush=True)
    elif isinstance(summary, dict):
        print(_format_block(labels[0] if labels else "(flat)", summary), flush=True)

    if dry_run:
        print(f"  [dry-run] not writing to {dest}", flush=True)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  wrote: {dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="One or more metrics_repeated_splits.json files to backfill.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional destination for a single input JSON (allows reading a "
            "renamed backup and writing back to the canonical filename). "
            "Ignored when more than one input is passed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but do not write any file.",
    )
    parser.add_argument(
        "--also-sqrt-grand-mean",
        action="store_true",
        help=(
            "Also record sqrt(mse_mean) as ``rmse_sqrt_of_mean_mse`` in each "
            "summary block. Differs from rmse_mean by Jensen's inequality; "
            "use only if you specifically want the 'global RMSE' interpretation."
        ),
    )
    args = parser.parse_args()

    if args.out is not None and len(args.inputs) != 1:
        print(
            "--out is only valid with a single input file; got "
            f"{len(args.inputs)}. Run the script once per file instead.",
            file=sys.stderr,
        )
        sys.exit(2)

    for src in args.inputs:
        dest = args.out if (args.out is not None and len(args.inputs) == 1) else src
        process_file(
            src,
            dest,
            also_sqrt_grand_mean=args.also_sqrt_grand_mean,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
