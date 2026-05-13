"""Corrected paired Student's t-test (Nadeau & Bengio, 2003) for the
repeated-splits benchmark.

For every model in the saved repeated-splits JSONs, runs a corrected paired
t-test against ``baseline:train_mean`` on each per-split metric (MSE, MAE,
R²). The correction inflates the naive SE to account for the dependence
between resampled splits that re-use the same underlying rows:

    SE_corrected² = σ²_diff × (1 / n_splits + n_test / n_train)

where ``σ²_diff`` is the sample variance of the per-split paired differences,
``n_splits`` is the number of resamples (here 500), and ``n_test`` /
``n_train`` are the holdout / train sizes used in each split. Degrees of
freedom = ``n_splits − 1``.

Reference:
    Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
    Machine Learning, 52(3), 239–281.

Run::

    python -m scripts.corrected_paired_ttest

Or with explicit JSONs / alpha::

    python -m scripts.corrected_paired_ttest \
        --sklearn-json checkpoints/modeling/metrics_repeated_splits.json \
        --bert-json checkpoints/bert_propa/metrics_repeated_splits.json \
        --alpha 0.05 \
        --out checkpoints/modeling/corrected_paired_ttest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402  (import after sys.path edit)
    BERT_CHECKPOINT_DIR,
    HOLDOUT_N,
    METRICS_REPEATED_SPLITS_JSON,
)

BASELINE_KEY = "baseline:train_mean"
METRICS: tuple[tuple[str, str, str], ...] = (
    # (key in per_seed_row, human label, direction: "lower" or "higher" is better)
    ("mse", "MSE", "lower"),
    ("rmse", "RMSE", "lower"),
    ("mae", "MAE", "lower"),
    ("r2_vs_const", "R²", "higher"),
)


def _load_seed_metric_map(rows: list[dict]) -> dict[int, dict]:
    """Index a per-seed list by seed for safe seed-aligned pairing."""
    out: dict[int, dict] = {}
    for r in rows:
        if "seed" not in r:
            continue
        out[int(r["seed"])] = r
    return out


def _collect_models(
    sklearn_json: Path, bert_json: Path | None
) -> tuple[dict[str, dict[int, dict]], dict, dict]:
    """Return ``{model_name: {seed: row}}`` plus parsed metadata for both JSONs."""
    if not sklearn_json.is_file():
        raise SystemExit(
            f"Missing repeated-splits JSON: {sklearn_json}\n"
            "Run `python run_model_benchmark.py repeated-splits` first."
        )
    sk = json.loads(sklearn_json.read_text(encoding="utf-8"))
    per_seed = sk.get("per_seed_rows") or {}
    if not isinstance(per_seed, dict):
        raise SystemExit(
            f"Unexpected sklearn JSON shape: per_seed_rows is {type(per_seed).__name__}, "
            "expected dict of model -> list[row]."
        )
    models: dict[str, dict[int, dict]] = {
        name: _load_seed_metric_map(rows) for name, rows in per_seed.items()
    }
    if BASELINE_KEY not in models:
        raise SystemExit(
            f"`{BASELINE_KEY}` not present in {sklearn_json}. "
            "Re-run `repeated-splits` to regenerate with the train-mean baseline."
        )

    if bert_json is not None and bert_json.is_file():
        be = json.loads(bert_json.read_text(encoding="utf-8"))
        bert_rows = be.get("per_seed_rows") or []
        if not isinstance(bert_rows, list):
            raise SystemExit(
                f"Unexpected BERT JSON shape: per_seed_rows is {type(bert_rows).__name__}, "
                "expected flat list[row]."
            )
        bert_label = f"bert:{be.get('model_id', 'unknown')}"
        models[bert_label] = _load_seed_metric_map(bert_rows)
        bert_meta = {
            "path": str(bert_json),
            "n_repeats": int(be.get("n_repeats", len(bert_rows))),
            "base_seed": be.get("base_seed"),
            "label": bert_label,
        }
    else:
        bert_meta = {}

    sklearn_meta = {
        "path": str(sklearn_json),
        "n_repeats": int(sk.get("n_repeats", 0)),
        "base_seed": sk.get("base_seed"),
        "holdout_n": int(sk.get("holdout_n", HOLDOUT_N)),
    }
    return models, sklearn_meta, bert_meta


def corrected_paired_t(
    diff: np.ndarray, n_train: int, n_test: int
) -> tuple[float, float, float, int]:
    """Nadeau-Bengio corrected paired t statistic + two-sided p-value.

    Returns ``(t_stat, p_two_sided, d_bar, df)``. NaNs in ``diff`` are dropped.
    """
    d = diff[~np.isnan(diff)]
    n_splits = int(d.size)
    if n_splits < 2:
        return float("nan"), float("nan"), float("nan"), 0
    d_bar = float(np.mean(d))
    # ``ddof=1`` for the unbiased sample variance, consistent with the
    # canonical Nadeau-Bengio derivation.
    sigma2 = float(np.var(d, ddof=1))
    correction = (1.0 / n_splits) + (n_test / float(n_train))
    se = float(np.sqrt(sigma2 * correction))
    if se == 0.0 or not np.isfinite(se):
        return float("nan"), float("nan"), d_bar, n_splits - 1
    t_stat = d_bar / se
    df = n_splits - 1
    p_two = float(2.0 * student_t.sf(abs(t_stat), df))
    return float(t_stat), p_two, d_bar, df


def _sig_marker(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return "ns"


def _paired_diff(
    model_rows: dict[int, dict],
    baseline_rows: dict[int, dict],
    metric_key: str,
    *,
    higher_is_better: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned (baseline_vals, model_vals, diff) for the metric.

    ``diff`` is oriented so that **positive means the model beats the baseline**,
    i.e. ``baseline - model`` for "lower is better" metrics and ``model - baseline``
    for "higher is better" metrics.
    """
    seeds = sorted(set(model_rows.keys()) & set(baseline_rows.keys()))
    if not seeds:
        return np.array([]), np.array([]), np.array([])
    baseline_vals = np.array(
        [baseline_rows[s].get(metric_key, np.nan) for s in seeds], dtype=float
    )
    model_vals = np.array(
        [model_rows[s].get(metric_key, np.nan) for s in seeds], dtype=float
    )
    diff = (model_vals - baseline_vals) if higher_is_better else (baseline_vals - model_vals)
    return baseline_vals, model_vals, diff


def run_table(
    models: dict[str, dict[int, dict]],
    *,
    n_train: int,
    n_test: int,
    alpha: float,
) -> pd.DataFrame:
    baseline_rows = models[BASELINE_KEY]
    records: list[dict] = []
    other_models = [m for m in models.keys() if m != BASELINE_KEY]
    for metric_key, metric_label, direction in METRICS:
        higher_is_better = direction == "higher"
        base_arr = np.array(
            [r.get(metric_key, np.nan) for r in baseline_rows.values()], dtype=float
        )
        base_mean = float(np.nanmean(base_arr))
        for model_name in other_models:
            baseline_vals, model_vals, diff = _paired_diff(
                models[model_name],
                baseline_rows,
                metric_key,
                higher_is_better=higher_is_better,
            )
            if diff.size == 0:
                continue
            t_stat, p_two, d_bar, df = corrected_paired_t(diff, n_train, n_test)
            model_mean = float(np.nanmean(model_vals))
            records.append(
                {
                    "metric": metric_label,
                    "direction": direction,
                    "model": model_name,
                    "baseline_mean": base_mean,
                    "model_mean": model_mean,
                    "delta_mean_(model_better_if_positive)": d_bar,
                    "t_stat": t_stat,
                    "df": df,
                    "p_two_sided": p_two,
                    "significant_at_alpha": p_two < alpha if np.isfinite(p_two) else False,
                    "marker": _sig_marker(p_two, alpha),
                    "n_paired_splits": int(diff.size),
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sklearn-json",
        type=Path,
        default=METRICS_REPEATED_SPLITS_JSON,
        help="repeated-splits JSON for LR/RF/LLM + baseline.",
    )
    parser.add_argument(
        "--bert-json",
        type=Path,
        default=BERT_CHECKPOINT_DIR / "metrics_repeated_splits.json",
        help="repeated-splits JSON for BERT (optional; skipped if missing).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)."
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Override train size (default: total rows − holdout_n).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Override test size (default: holdout_n from JSON, fallback config.HOLDOUT_N).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for the full results table.",
    )
    args = parser.parse_args()

    models, sk_meta, be_meta = _collect_models(args.sklearn_json, args.bert_json)
    holdout_n = sk_meta["holdout_n"]
    n_repeats = sk_meta["n_repeats"]
    n_train = args.n_train if args.n_train is not None else max(
        len(models[BASELINE_KEY]) - holdout_n,
        # Repeated-splits saves seeds, not the total table size; fall back to
        # the documented contract: 500 rows total, 100 held out → 400 train.
        500 - holdout_n,
    )
    n_test = args.n_test if args.n_test is not None else holdout_n

    print(
        f"Corrected paired t-test (Nadeau & Bengio, 2003)\n"
        f"  sklearn json: {sk_meta['path']}\n"
        + (f"  bert json:    {be_meta['path']}\n" if be_meta else "  bert json:    (not provided)\n")
        + f"  n_splits (k) = {n_repeats}   df = {n_repeats - 1}\n"
        f"  n_train = {n_train}   n_test = {n_test}\n"
        f"  correction factor = 1/{n_repeats} + {n_test}/{n_train} = "
        f"{1.0 / n_repeats + n_test / n_train:.6f}\n"
        f"  baseline = {BASELINE_KEY!r}   alpha = {args.alpha}\n",
        flush=True,
    )

    df = run_table(models, n_train=n_train, n_test=n_test, alpha=args.alpha)
    if df.empty:
        print("No comparisons produced (no overlapping seeds?).")
        return

    show_cols = [
        "metric",
        "model",
        "baseline_mean",
        "model_mean",
        "delta_mean_(model_better_if_positive)",
        "t_stat",
        "df",
        "p_two_sided",
        "marker",
        "n_paired_splits",
    ]
    fmt = df[show_cols].copy()
    fmt["baseline_mean"] = fmt["baseline_mean"].map(lambda v: f"{v:.4f}")
    fmt["model_mean"] = fmt["model_mean"].map(lambda v: f"{v:.4f}")
    fmt["delta_mean_(model_better_if_positive)"] = fmt[
        "delta_mean_(model_better_if_positive)"
    ].map(lambda v: f"{v:+.4f}")
    fmt["t_stat"] = fmt["t_stat"].map(
        lambda v: f"{v:+.3f}" if np.isfinite(v) else "nan"
    )
    fmt["p_two_sided"] = fmt["p_two_sided"].map(
        lambda v: f"{v:.3e}" if np.isfinite(v) else "nan"
    )

    out_lines: list[str] = []
    for _, metric_label, _ in METRICS:
        block = fmt[fmt["metric"] == metric_label]
        if block.empty:
            continue
        out_lines.append(f"=== {metric_label} ===")
        out_lines.append(block.drop(columns=["metric"]).to_string(index=False))
        out_lines.append("")
    out_lines.append("Legend: *** p<0.001  ** p<0.01  * p<alpha  ns not significant")
    out_lines.append(
        "Sign convention: positive delta means model beats baseline "
        "(lower MSE/MAE or higher R²)."
    )
    sys.stdout.write("\n".join(out_lines) + "\n")
    sys.stdout.flush()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nWrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()
