"""Repeated random 400/100 splits for LR + RF (+ optional LLM variants).

Robustness CI that complements the fixed-holdout bootstrap: re-draws a fresh
stratified split every iteration and refits the tabular models with their saved
best params (no inner grid). LLM variants are not refit — they have predictions
for all 500 modeling rows from ``cmd_llm``'s JSONL checkpoint, so each iteration
just slices those preds with the same ``idx_te`` used for LR/RF, capturing
test-set variability for LLM (vs. test-set + retrain variability for LR/RF).

Seed schedule is ``range(base_seed, base_seed + n_repeats)``; using the same
``base_seed`` here and in ``modeling.bert_repeated_splits`` guarantees BERT sees
the exact same per-iteration splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modeling.dataset import normalize_channel_column
from modeling.sklearn_models import make_pipeline


def split_indices(
    n: int,
    seed: int,
    holdout_n: int,
    strat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split with graceful fallback to a plain shuffle."""
    try:
        return train_test_split(
            np.arange(n),
            test_size=holdout_n,
            random_state=seed,
            stratify=strat,
        )
    except ValueError:
        return train_test_split(
            np.arange(n),
            test_size=holdout_n,
            random_state=seed,
            shuffle=True,
        )


def score_row(seed: int, y_te: np.ndarray, pred: np.ndarray) -> dict:
    """One per-split metric row; NaN predictions are dropped before scoring."""
    pred = np.asarray(pred, dtype=float)
    y_te = np.asarray(y_te, dtype=float)
    mask = ~np.isnan(pred)
    if int(mask.sum()) < 2:
        return {
            "seed": int(seed),
            "n": int(mask.sum()),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2_vs_const": float("nan"),
        }
    yt, pr = y_te[mask], pred[mask]
    diff = yt - pr
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse)) if mse >= 0 else float("nan")
    var_y = float(np.var(yt))
    r2 = (1.0 - mse / var_y) if var_y > 0 else float("nan")
    return {
        "seed": int(seed),
        "n": int(mask.sum()),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_vs_const": r2,
    }


def load_llm_aligned(
    table_df: pd.DataFrame, jsonl_path: Path
) -> np.ndarray:
    """Return a length-``len(table_df)`` array of LLM preds aligned by (channel, telegram_id).

    Missing rows (or non-numeric predictions) become ``NaN`` so downstream scoring
    can drop them per-split via ``score_row``.
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return np.full(len(table_df), np.nan, dtype=float)
    pred_df = pd.DataFrame(records)
    if "pred" not in pred_df.columns or "telegram_id" not in pred_df.columns:
        return np.full(len(table_df), np.nan, dtype=float)
    pred_df = pred_df[[c for c in ("channel", "telegram_id", "pred") if c in pred_df.columns]].copy()
    pred_df["pred"] = pd.to_numeric(pred_df["pred"], errors="coerce")
    pred_df["telegram_id"] = pd.to_numeric(pred_df["telegram_id"], errors="coerce").astype("Int64")
    base = normalize_channel_column(table_df[["channel", "telegram_id"]].copy())
    base["telegram_id"] = pd.to_numeric(base["telegram_id"], errors="coerce").astype("Int64")
    if "channel" in pred_df.columns:
        pred_df = normalize_channel_column(pred_df)
        merged = base.merge(pred_df, on=["channel", "telegram_id"], how="left")
    else:
        merged = base.merge(pred_df, on="telegram_id", how="left")
    return merged["pred"].to_numpy(dtype=float)


CONSTANT_TRAIN_MEAN_KEY = "baseline:train_mean"


def repeated_splits(
    X: np.ndarray,
    y: np.ndarray,
    rf_best_params: dict,
    llm_preds_by_name: dict[str, np.ndarray] | None = None,
    *,
    n_repeats: int = 500,
    holdout_n: int = 100,
    base_seed: int = 100,
    strat: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    """Run LR + RF (+ each LLM variant) + the per-split constant ``train-mean``
    baseline over ``n_repeats`` fresh stratified splits.

    Same seed schedule => same indices for every model on a given iteration.
    The constant baseline predicts ``mean(y_train_i)`` for every test row of
    split ``i``; its CI captures how much the baseline itself shifts with the
    split, so model vs. baseline comparisons stay apples-to-apples.
    """
    out: dict[str, list[dict]] = {CONSTANT_TRAIN_MEAN_KEY: [], "lr": [], "rf": []}
    llm_preds_by_name = llm_preds_by_name or {}
    for name in llm_preds_by_name:
        out[name] = []
    n = len(X)
    for i in range(n_repeats):
        s = base_seed + i
        idx_tr, idx_te = split_indices(n, s, holdout_n, strat)
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]

        c_train = float(np.mean(y_tr))
        out[CONSTANT_TRAIN_MEAN_KEY].append(
            score_row(s, y_te, np.full(len(y_te), c_train, dtype=float))
        )

        pipe_lr = make_pipeline(LinearRegression(), scale=True).fit(X_tr, y_tr)
        out["lr"].append(score_row(s, y_te, pipe_lr.predict(X_te)))

        pipe_rf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        max_features="sqrt",
                        random_state=s,
                        n_jobs=-1,
                        **rf_best_params,
                    ),
                ),
            ]
        ).fit(X_tr, y_tr)
        out["rf"].append(score_row(s, y_te, pipe_rf.predict(X_te)))

        for name, preds in llm_preds_by_name.items():
            out[name].append(score_row(s, y_te, preds[idx_te]))
    return out


def summarize_repeats(rows: list[dict]) -> dict:
    """Mean + 2.5/97.5 percentile CIs for MSE / RMSE / MAE / R² across all split rows.

    ``rmse_mean`` is the **mean of per-seed RMSEs** (not ``sqrt(mse_mean)``); this
    keeps RMSE on the same footing as MSE/MAE/R² where every summary stat comes
    from the same per-seed distribution. RMSE for legacy rows that lack it falls
    back to ``sqrt(mse)`` on read.
    """
    if not rows:
        return {
            "n_repeats": 0,
            "mean_n_per_split": 0.0,
            "mse_mean": float("nan"),
            "rmse_mean": float("nan"),
            "mae_mean": float("nan"),
            "r2_mean": float("nan"),
            "mse_ci_low": float("nan"),
            "mse_ci_high": float("nan"),
            "rmse_ci_low": float("nan"),
            "rmse_ci_high": float("nan"),
            "mae_ci_low": float("nan"),
            "mae_ci_high": float("nan"),
            "r2_ci_low": float("nan"),
            "r2_ci_high": float("nan"),
        }
    mse = np.array([r.get("mse", np.nan) for r in rows], dtype=float)
    rmse = np.array(
        [r.get("rmse", np.sqrt(r["mse"]) if r.get("mse") is not None and r["mse"] >= 0 else np.nan)
         for r in rows],
        dtype=float,
    )
    mae = np.array([r.get("mae", np.nan) for r in rows], dtype=float)
    r2 = np.array([r.get("r2_vs_const", np.nan) for r in rows], dtype=float)
    n_arr = np.array([r.get("n", 0) for r in rows], dtype=float)
    return {
        "n_repeats": int(len(rows)),
        "mean_n_per_split": float(np.mean(n_arr)),
        "mse_mean": float(np.nanmean(mse)),
        "rmse_mean": float(np.nanmean(rmse)),
        "mae_mean": float(np.nanmean(mae)),
        "r2_mean": float(np.nanmean(r2)),
        "mse_ci_low": float(np.nanquantile(mse, 0.025)),
        "mse_ci_high": float(np.nanquantile(mse, 0.975)),
        "rmse_ci_low": float(np.nanquantile(rmse, 0.025)),
        "rmse_ci_high": float(np.nanquantile(rmse, 0.975)),
        "mae_ci_low": float(np.nanquantile(mae, 0.025)),
        "mae_ci_high": float(np.nanquantile(mae, 0.975)),
        "r2_ci_low": float(np.nanquantile(r2, 0.025)),
        "r2_ci_high": float(np.nanquantile(r2, 0.975)),
    }
