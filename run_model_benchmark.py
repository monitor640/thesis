#!/usr/bin/env python3
"""
Inter-rater reliability + a small, thesis-friendly model stack (see ``config.py`` for paths).

**Evaluation contract (what to cite in the thesis)**

1. **Tabular (sklearn)** — Two baselines: **OLS linear regression** (fit on the train pool only,
   evaluated on the fixed **100-row holdout**) and **one random forest** chosen by **GridSearchCV**
   with inner 5-fold CV on the train pool, then the same holdout. No separate outer CV loop;
   metrics JSON records holdout errors only (plus RF ``best_params_`` / inner CV score).
   Same holdout manifest is used for BERT and for the LLM holdout line.
   Fitted models are not saved; ``sklearn-importances`` refits LR+RF on the train pool only (no RF grid)
   using ``metrics_sklearn.json`` to export linear coefficients and RF ``feature_importances_``.

2. **LLM** — One run scores **every row in the modeling table** (typically 500) that merges to
   labels + predictions. Metrics JSON includes ``regression_full_table`` (all merged rows) and
   ``holdout`` (the 100 rows in the manifest), so you can report both without a second API pass.

3. **BERT** — One fine-tuned sequence model (default ``BERT_MODEL_ID`` in config, e.g. XLM-R);
   evaluated on the same 100-row holdout after training on the train pool.

4. **Tüüp** — Optional auxiliary task (TF–IDF + RF on propaganda type); not part of the 0–4 stack.

Pipeline order: export-table → reliability → sklearn → ``llm`` ×2 (``no_features``, ``with_features``)
→ tyyp → bert. LLM needs ``OPENAI_API_KEY``; optional ``OPENAI_MODEL`` / ``LLM_MODEL``.

**Robustness CI (separate manual step):** ``repeated-splits`` re-draws 500 fresh
stratified 400/100 splits and refits LR + RF with saved ``rf_gridsearch.best_params``
on each one; LLM predictions are sliced from the cached JSONL by the same
``idx_te``. ``bert-repeated-splits`` does the same for BERT using saved
``final_*`` hparams (no inner grid search; GPU strongly recommended). Both are
intentionally excluded from ``all`` so a default run does not trigger hours of
training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import (
    BASE_DIR,
    BERT_CHECKPOINT_DIR,
    HOLDOUT_MANIFEST_CSV,
    HOLDOUT_N,
    HOLDOUT_RANDOM_STATE,
    HOLDOUT_SKLEARN_PREDS_CSV,
    LABELING_WITH_FEATURES_CSV,
    LLM_CHECKPOINT_DIR,
    llm_checkpoint_jsonl,
    metrics_llm_json_path,
    MODEL_COMPARISON_CSV,
    MODELING_TABLE_CSV,
    METRICS_RELIABILITY_JSON,
    METRICS_REPEATED_SPLITS_JSON,
    METRICS_SKLEARN_JSON,
    METRICS_TYYP_JSON,
    SIMON_LABELING_CSV,
    SKLEARN_CV_FOLDS,
    UKU_LABELING_CSV,
)
from modeling.dataset import (
    build_modeling_table,
    normalize_channel_column,
    validate_modeling_table,
)
from modeling.feature_config import (
    feature_selection_rule,
    load_modeling_features_from_config,
    resolve_feature_columns,
)
from modeling.holdout_llm_csvs import write_holdout_llm_csv_from_merged
from modeling.llm_predict import default_model, predict_propa_llm
from modeling.metrics import evaluate_multiclass, evaluate_regression, write_metrics_json
from modeling.reliability import agreement_report
from modeling.repeated_splits import (
    load_llm_aligned,
    repeated_splits,
    summarize_repeats,
)
from modeling.sklearn_models import fit_predict
from modeling.tyyp_sklearn import cv_tyyp_tfidf_rf


_REG_METRIC_KEYS = (
    "mae",
    "mse",
    "rmse",
    "accuracy_rounded_0_4",
    "var_y_true",
    "r2_vs_const",
    "mse_ci_low",
    "mse_ci_high",
    "mae_ci_low",
    "mae_ci_high",
    "r2_ci_low",
    "r2_ci_high",
    "ci_n_resamples",
    "report",
)


def _reg_metrics_dict(m: dict) -> dict:
    """Numeric + CI + report string from :func:`evaluate_regression`."""
    return {k: m[k] for k in _REG_METRIC_KEYS if k in m}


def _llm_usage_totals(pred_df) -> dict | None:
    """Sum OpenAI ``usage`` dicts from ``predict_propa_llm`` output, if present."""
    if pred_df is None or "usage" not in pred_df.columns:
        return None
    sp = sc = st = 0
    n = 0
    for u in pred_df["usage"]:
        if isinstance(u, dict):
            sp += int(u.get("prompt_tokens", 0) or 0)
            sc += int(u.get("completion_tokens", 0) or 0)
            st += int(u.get("total_tokens", 0) or 0)
            n += 1
    if n == 0:
        return None
    return {
        "rows_with_usage": n,
        "prompt_tokens": sp,
        "completion_tokens": sc,
        "total_tokens": st,
    }


def _cls_metrics_dict(m: dict) -> dict:
    out = {
        k: m[k]
        for k in ("accuracy", "macro_f1", "weighted_f1")
        if k in m
    }
    if "report" in m:
        out["report"] = m["report"]
    return out


def cmd_reliability(args: argparse.Namespace) -> None:
    import pandas as pd

    from modeling.dataset import add_normalized_channel

    uku = add_normalized_channel(pd.read_csv(args.uku, encoding="utf-8-sig"))
    simon = add_normalized_channel(pd.read_csv(args.simon, encoding="utf-8-sig"))
    uku = uku.rename(columns={"Propa": "propa_uku"})
    simon = simon.rename(columns={"Propa": "propa_simon"})
    for df in (uku, simon):
        df["_tid"] = pd.to_numeric(df["telegram_id"], errors="coerce").astype("Int64")
    m = uku.merge(
        simon[["channel", "_tid", "propa_simon"]],
        on=["channel", "_tid"],
        how="inner",
    )
    rep = agreement_report(m["propa_uku"], m["propa_simon"])
    print(json.dumps(rep, indent=2))
    write_metrics_json(
        METRICS_RELIABILITY_JSON,
        {
            "schema": "benchmark_reliability_v1",
            "uku": str(args.uku),
            "simon": str(args.simon),
            **rep,
        },
    )


def cmd_export_table(args: argparse.Namespace) -> None:
    df = build_modeling_table(args.uku, args.simon, args.features)
    out = MODELING_TABLE_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows to {out}")
    if args.features:
        n_feat = int(df["id"].notna().sum())
        if n_feat < len(df):
            print(
                "Warning: not all rows have merged features — regenerate labeling_with_features "
                "from the same export as the labeling sheet."
            )


def _feature_names(df) -> list[str]:
    names = load_modeling_features_from_config()
    return resolve_feature_columns(df, names)


def _split_train_holdout(
    n: int, y_strat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 on rounded 0–4 when possible; falls back to a plain shuffle."""
    from sklearn.model_selection import train_test_split

    try:
        return train_test_split(
            np.arange(n),
            test_size=HOLDOUT_N,
            random_state=HOLDOUT_RANDOM_STATE,
            stratify=y_strat,
        )
    except ValueError:
        return train_test_split(
            np.arange(n),
            test_size=HOLDOUT_N,
            random_state=HOLDOUT_RANDOM_STATE,
            shuffle=True,
        )


def make_tuned_rf(random_state: int):
    """5×5 inner CV over RF hyperparameters; refits best model on full training matrix when fitted."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, KFold

    base_rf = RandomForestRegressor(
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [100,200,300],
        "max_depth": [4,6,8],
        "min_samples_leaf": [3, 5, 7, 9, 11],
    }
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    return GridSearchCV(
        base_rf,
        param_grid,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


# Exactly two tabular models (thesis narrative): OLS linear baseline + one RF (RF tuned via GridSearchCV).
def cmd_sklearn(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_csv(args.table, encoding="utf-8-sig")
    try:
        validate_modeling_table(df, need_text=False)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    feats = _feature_names(df)
    if not feats:
        raise SystemExit("No feature columns after config + tests CSV filter.")

    y_raw = pd.to_numeric(df["y_propa"], errors="coerce")
    df = df.loc[y_raw.notna()].reset_index(drop=True)
    y = y_raw.loc[y_raw.notna()].values.astype(float)
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if len(df) <= HOLDOUT_N:
        raise SystemExit(
            f"Need more than {HOLDOUT_N} rows for {HOLDOUT_N}-row holdout + training."
        )

    print(
        "Target: continuous consensus (y_propa); linear (holdout only) + RF "
        "(GridSearchCV with inner 5-fold on train pool, then holdout).",
        flush=True,
    )
    print(
        f"{len(df) - HOLDOUT_N} train / {HOLDOUT_N} holdout (seed {HOLDOUT_RANDOM_STATE}; "
        f"see config). Manifest: {HOLDOUT_MANIFEST_CSV.name}"
    )
    print(
        "Stratified split on rounded 0-4 when possible; else random shuffle with the same seed."
    )

    idx_tr, idx_te = _split_train_holdout(len(df), np.rint(y).clip(0, 4).astype(int))
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    man_cols = ["channel", "telegram_id"]
    HOLDOUT_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[idx_te][man_cols].to_csv(
        HOLDOUT_MANIFEST_CSV, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote holdout manifest ({len(idx_te)} rows): {HOLDOUT_MANIFEST_CSV}")

    holdout_metrics: dict[str, dict] = {}
    holdout_preds: dict[str, np.ndarray] = {}

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    print("\n=== linear: OLS on train pool → predict holdout ===", flush=True)
    pred_lin = fit_predict(X_tr, y_tr, X_te, LinearRegression(), scale=True)
    m_lin = evaluate_regression(y_te, pred_lin)
    print(m_lin["report"], flush=True)
    holdout_metrics["linear"] = m_lin
    holdout_preds["linear"] = pred_lin

    print(
        "\n=== rf: Pipeline(imputer, GridSearchCV) on train pool → predict holdout ===",
        flush=True,
    )
    pipe_rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("grid", make_tuned_rf(HOLDOUT_RANDOM_STATE)),
        ]
    )
    pipe_rf.fit(X_tr, y_tr)
    pred_rf = pipe_rf.predict(X_te)
    m_rf = evaluate_regression(y_te, pred_rf)
    print(m_rf["report"], flush=True)
    grid = pipe_rf.named_steps["grid"]
    print("RF best_params:", grid.best_params_, flush=True)
    print(
        "RF inner-CV best score (neg_mean_squared_error):",
        grid.best_score_,
        flush=True,
    )
    holdout_metrics["rf"] = m_rf
    holdout_preds["rf"] = pred_rf

    out = df.iloc[idx_te][man_cols].copy()
    out["y_true"] = y_te
    for name, pred in holdout_preds.items():
        out[f"pred_{name}"] = pred
    HOLDOUT_SKLEARN_PREDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(HOLDOUT_SKLEARN_PREDS_CSV, index=False, encoding="utf-8-sig")
    print("Wrote holdout predictions:", HOLDOUT_SKLEARN_PREDS_CSV)

    write_metrics_json(
        METRICS_SKLEARN_JSON,
        {
            "schema": "benchmark_sklearn_v5",
            "table": str(args.table.resolve()),
            "n_rows_labeled": int(len(df)),
            "n_train": int(len(idx_tr)),
            "n_holdout": int(len(idx_te)),
            "random_state": HOLDOUT_RANDOM_STATE,
            "feature_selection_rule": feature_selection_rule(),
            "feature_selection_source": "preliminary_propa_group_tests.csv (binary 500 sample, disjoint from modeling 500)",
            "feature_count": len(feats),
            "features": feats,
            "models": ["linear", "rf"],
            "holdout": {name: _reg_metrics_dict(m) for name, m in holdout_metrics.items()},
            "rf_gridsearch": {
                "inner_cv_folds": 5,
                "scoring": "neg_mean_squared_error",
                "best_params": grid.best_params_,
                "best_neg_mean_squared_error": float(grid.best_score_),
            },
            "holdout_manifest_csv": str(HOLDOUT_MANIFEST_CSV),
            "holdout_predictions_csv": str(HOLDOUT_SKLEARN_PREDS_CSV),
        },
    )


def cmd_sklearn_importances(args: argparse.Namespace) -> None:
    """Refit LR + RF on the train split only (no RF grid search) using ``metrics_sklearn.json``; write importances.

    Fitted estimators were never pickled, so coefficients / ``feature_importances_`` cannot be read from disk
    alone. This replays the same preprocessing and split as ``sklearn`` and fits once with the saved RF
    ``best_params``—seconds of CPU, identical models if the modeling table and sklearn version match.

    Only ``--out`` is written (default: ``sklearn_feature_importances.json`` next to ``--metrics``).
    ``--metrics`` (e.g. ``metrics_sklearn.json``) is read-only unless you set ``--out`` to that same path.
    """
    import json

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from modeling.sklearn_models import make_pipeline

    metrics_path = Path(args.metrics)
    if not metrics_path.is_file():
        raise SystemExit(f"Metrics JSON not found: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as f:
        meta = json.load(f)

    feats = meta.get("features")
    if not feats or not isinstance(feats, list):
        raise SystemExit(f"Missing 'features' list in {metrics_path}")

    rf_block = meta.get("rf_gridsearch") or {}
    best_params = rf_block.get("best_params")
    if not best_params:
        raise SystemExit(f"Missing rf_gridsearch.best_params in {metrics_path}")

    rs = int(meta.get("random_state", HOLDOUT_RANDOM_STATE))

    df = pd.read_csv(args.table, encoding="utf-8-sig")
    try:
        validate_modeling_table(df, need_text=False)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    resolved = _feature_names(df)
    if resolved != list(feats):
        raise SystemExit(
            "Feature columns from the table do not match metrics JSON.\n"
            f"  metrics ({len(feats)}): {feats}\n"
            f"  table   ({len(resolved)}): {resolved}\n"
            "Regenerate the table / metrics with the same feature config, or edit the JSON."
        )

    y_raw = pd.to_numeric(df["y_propa"], errors="coerce")
    df = df.loc[y_raw.notna()].reset_index(drop=True)
    y = y_raw.loc[y_raw.notna()].values.astype(float)
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    idx_tr, idx_te = _split_train_holdout(len(df), np.rint(y).clip(0, 4).astype(int))
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]

    pipe_lin = make_pipeline(LinearRegression(), scale=True).fit(X_tr, y_tr)
    est = pipe_lin.named_steps["est"]
    scaler = pipe_lin.named_steps["scaler"]
    coef_scaled = np.asarray(est.coef_, dtype=float).ravel()
    scaler_mean = np.asarray(scaler.mean_, dtype=float)
    scaler_scale = np.asarray(scaler.scale_, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        coef_raw_unit = np.divide(
            coef_scaled,
            scaler_scale,
            out=np.full_like(coef_scaled, np.nan),
            where=np.abs(scaler_scale) > 1e-12,
        )

    pipe_rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    max_features="sqrt",
                    random_state=rs,
                    n_jobs=-1,
                    **best_params,
                ),
            ),
        ]
    )
    pipe_rf.fit(X_tr, y_tr)
    rf = pipe_rf.named_steps["rf"]
    imp = np.asarray(rf.feature_importances_, dtype=float)

    pred_lin_te = pipe_lin.predict(X_te)
    pred_rf_te = pipe_rf.predict(X_te)
    mse_lin_re = float(np.mean((y_te - pred_lin_te) ** 2))
    mse_rf_re = float(np.mean((y_te - pred_rf_te) ** 2))

    hold_block = meta.get("holdout") or {}
    lin_h = hold_block.get("linear") or {}
    rf_h = hold_block.get("rf") or {}
    if "mse" not in lin_h or "mse" not in rf_h:
        raise SystemExit("metrics JSON missing holdout.linear['mse'] or holdout.rf['mse']")
    mse_lin_rep = float(lin_h["mse"])
    mse_rf_rep = float(rf_h["mse"])

    abs_tol = float(args.sanity_mse_atol)
    ok_lin = abs(mse_lin_re - mse_lin_rep) <= abs_tol
    ok_rf = abs(mse_rf_re - mse_rf_rep) <= abs_tol

    print("\n=== sanity: holdout MSE (refit vs metrics JSON) ===", flush=True)
    print(
        f"  linear  refit={mse_lin_re:.12f}  reported={mse_lin_rep:.12f}  "
        f"|Δ|={abs(mse_lin_re - mse_lin_rep):.3e}  pass={ok_lin}",
        flush=True,
    )
    print(
        f"  rf      refit={mse_rf_re:.12f}  reported={mse_rf_rep:.12f}  "
        f"|Δ|={abs(mse_rf_re - mse_rf_rep):.3e}  pass={ok_rf}",
        flush=True,
    )
    if not ok_lin or not ok_rf:
        print(
            f"\nSanity check failed (|Δ| must be ≤ {abs_tol:g}). "
            "Different sklearn build, table row order, or benchmark code can cause this.",
            flush=True,
        )
        if not args.allow_sanity_fail:
            raise SystemExit(1)
    else:
        print("  Both models: holdout MSE matches metrics JSON within tolerance.", flush=True)

    out_path = Path(args.out) if args.out is not None else metrics_path.with_name("sklearn_feature_importances.json")
    payload = {
        "schema": "benchmark_sklearn_importances_v1",
        "source_metrics_json": str(metrics_path.resolve()),
        "modeling_table": str(Path(args.table).resolve()),
        "n_train_refit": int(len(idx_tr)),
        "n_holdout_refit": int(len(idx_te)),
        "random_state": rs,
        "features": list(feats),
        "sanity_holdout_mse": {
            "linear": {
                "refit": mse_lin_re,
                "reported": mse_lin_rep,
                "abs_diff": abs(mse_lin_re - mse_lin_rep),
                "passed": ok_lin,
            },
            "rf": {
                "refit": mse_rf_re,
                "reported": mse_rf_rep,
                "abs_diff": abs(mse_rf_re - mse_rf_rep),
                "passed": ok_rf,
            },
            "abs_tolerance": abs_tol,
            "exit_nonzero_on_fail": not bool(args.allow_sanity_fail),
        },
        "linear": {
            "intercept": float(est.intercept_),
            "coef_on_scaled_features": coef_scaled.tolist(),
            "approx_slope_per_raw_unit": np.nan_to_num(coef_raw_unit, nan=0.0).tolist(),
            "scaler_mean": scaler_mean.tolist(),
            "scaler_scale": scaler_scale.tolist(),
        },
        "rf": {
            "best_params": best_params,
            "feature_importances": imp.tolist(),
        },
    }
    write_metrics_json(out_path, payload)
    print(f"Wrote feature importances + linear coefficients: {out_path}", flush=True)
    print(
        "Note: linear ``coef_on_scaled_features`` match StandardScaler space; "
        "``approx_slope_per_raw_unit`` divides by ``scaler_scale`` (undefined → 0). "
        "``scaler_mean`` / ``scaler_scale`` are the per-feature StandardScaler stats fit on the train split.",
        flush=True,
    )


def _llm_truth_and_merge(df, pred_df):
    """Align LLM preds to human ``y_propa`` (table must pass :func:`validate_modeling_table`)."""
    import pandas as pd

    truth = normalize_channel_column(
        df[["channel", "telegram_id", "y_propa"]].rename(columns={"y_propa": "y_true"})
    )
    preds = pred_df.dropna(subset=["pred"]).copy()
    preds["pred"] = pd.to_numeric(preds["pred"], errors="coerce")
    preds = preds.dropna(subset=["pred"])
    # Checkpoints from before channel-aware keys may omit ``channel``; merge on tid only then.
    if "channel" in preds.columns:
        preds = normalize_channel_column(preds)
        m = truth.merge(preds, on=["channel", "telegram_id"], how="inner")
    else:
        m = truth.merge(preds, on="telegram_id", how="inner")
    m["y_true"] = pd.to_numeric(m["y_true"], errors="coerce")
    return m.dropna(subset=["y_true", "pred"])


def cmd_llm(args: argparse.Namespace) -> None:
    import pandas as pd

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise SystemExit(
            "LLM: missing OPENAI_API_KEY. Set it in "
            f"{BASE_DIR / '.env'} (or export in shell). "
            "If it is in .env but still missing, unset an empty OPENAI_API_KEY in the shell — "
            "that blocks dotenv from filling it."
        )

    df = pd.read_csv(args.table, encoding="utf-8-sig")
    try:
        validate_modeling_table(df, need_text=True)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    feats = _feature_names(df) if args.variant == "with_features" else None

    if getattr(args, "model", None):
        os.environ["OPENAI_MODEL"] = args.model
    model_id = default_model()
    ck = llm_checkpoint_jsonl(args.variant, model_id=model_id)
    print(
        f"[LLM {args.variant}] {len(df)} table rows → model={model_id!r}; "
        f"checkpoint={ck.name}",
        flush=True,
    )
    pred_df = predict_propa_llm(
        df,
        variant=args.variant,
        checkpoint_path=ck,
        feature_cols=feats or [],
    )

    m = _llm_truth_and_merge(df, pred_df)
    metrics_path = metrics_llm_json_path(args.variant, model_id=model_id)
    if len(m) == 0:
        print("No LLM predictions to score (check checkpoint / API).")
        write_metrics_json(
            metrics_path,
            {
                "schema": "benchmark_llm_v2",
                "variant": args.variant,
                "table": str(args.table.resolve()),
                "table_n_rows": int(len(df)),
                "model": model_id,
                "checkpoint_jsonl": str(ck.resolve()),
                "rows_scored_merged": 0,
                "note": "No predictions to score (empty merge or API/checkpoint issue).",
            },
        )
        return

    yt = m["y_true"].values.astype(float)
    pr = m["pred"].values.astype(float)
    reg_all = evaluate_regression(yt, pr)
    print("=== LLM vs y_propa — full modeling table (all merged rows) ===")
    print(f"Rows scored: {len(m)} (table has {len(df)} rows)")
    print(reg_all["report"])
    print(f"MSE (full table): {reg_all['mse']:.6f}")

    holdout_block: dict | None = None
    if HOLDOUT_MANIFEST_CSV.is_file():
        ho = normalize_channel_column(
            pd.read_csv(HOLDOUT_MANIFEST_CSV, encoding="utf-8-sig")
        )
        sub = m.merge(ho, on=["channel", "telegram_id"], how="inner")
        if len(sub) == 0:
            print("Holdout manifest merge: 0 rows (check channel + telegram_id).")
            holdout_block = {"n": 0, "note": "merge yielded 0 rows"}
        else:
            reg_h = evaluate_regression(
                sub["y_true"].values.astype(float),
                sub["pred"].values.astype(float),
            )
            print(f"=== Same LLM run: holdout test only (n={len(sub)}) ===")
            print(reg_h["report"])
            print(f"MSE (holdout): {reg_h['mse']:.6f}")
            holdout_block = {"n": int(len(sub)), **_reg_metrics_dict(reg_h)}
    else:
        print(f"(No holdout MSE: run sklearn first to create {HOLDOUT_MANIFEST_CSV})")
        holdout_block = None

    met = evaluate_multiclass(
        np.rint(yt).clip(0, 4).astype(int),
        np.rint(pr).clip(0, 4).astype(int),
    )
    print("=== Rounded 0–4 classification (secondary) ===")
    print(met["report"])
    print("macro_f1:", met["macro_f1"], "accuracy:", met["accuracy"])

    usage_block = _llm_usage_totals(pred_df)
    metrics_payload: dict = {
        "schema": "benchmark_llm_v2",
        "variant": args.variant,
        "table": str(args.table.resolve()),
        "table_n_rows": int(len(df)),
        "model": model_id,
        "checkpoint_jsonl": str(ck.resolve()),
        "rows_scored_merged": int(len(m)),
        "regression_full_table": _reg_metrics_dict(reg_all),
        "holdout": holdout_block,
        "multiclass_rounded_0_4": _cls_metrics_dict(met),
    }
    if usage_block is not None:
        metrics_payload["openai_usage"] = usage_block

    write_metrics_json(metrics_path, metrics_payload)

    write_holdout_llm_csv_from_merged(m, args.variant)


def cmd_tyyp(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_csv(args.table, encoding="utf-8-sig")
    try:
        validate_modeling_table(df, need_text=True, need_tyyp=True)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    texts = df["text"]
    y_tyyp = df["y_tyyp"]
    _, _, metrics, le = cv_tyyp_tfidf_rf(
        texts, y_tyyp, n_splits=SKLEARN_CV_FOLDS
    )
    print("Classes:", list(le.classes_))
    print(metrics["report"])
    print("macro_f1:", metrics["macro_f1"], "accuracy:", metrics["accuracy"])
    write_metrics_json(
        METRICS_TYYP_JSON,
        {
            "schema": "benchmark_tyyp_v1",
            "table": str(args.table.resolve()),
            "cv_folds": SKLEARN_CV_FOLDS,
            "classes": list(le.classes_),
            **_cls_metrics_dict(metrics),
        },
    )


def cmd_bert(_args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "modeling.bert_train"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


_COMPARE_COLUMNS = (
    "model",
    "eval_setup",
    "n",
    "mse",
    "mae",
    "rmse",
    "mse_ci_low",
    "mse_ci_high",
    "r2_vs_const",
    "r2_ci_low",
    "r2_ci_high",
    "accuracy_rounded_0_4",
    "source_file",
)


def _compare_row(
    model: str, eval_setup: str, n: int | None, m: dict, source: Path
) -> dict:
    return {
        "model": model,
        "eval_setup": eval_setup,
        "n": n,
        "mse": m.get("mse"),
        "mae": m.get("mae"),
        "rmse": m.get("rmse"),
        "mse_ci_low": m.get("mse_ci_low"),
        "mse_ci_high": m.get("mse_ci_high"),
        "r2_vs_const": m.get("r2_vs_const"),
        "r2_ci_low": m.get("r2_ci_low"),
        "r2_ci_high": m.get("r2_ci_high"),
        "accuracy_rounded_0_4": m.get("accuracy_rounded_0_4"),
        "source_file": str(source),
    }


def _collect_llm_aligned(df) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build {name -> length-len(df) array} for every LLM JSONL checkpoint on disk."""
    llm_preds: dict[str, np.ndarray] = {}
    jsonl_paths: list[str] = []
    for variant in ("no_features", "with_features"):
        for jp in sorted(LLM_CHECKPOINT_DIR.glob(f"llm_{variant}__*.jsonl")):
            stem = jp.stem
            sep = f"llm_{variant}__"
            model_slug = stem[len(sep) :] if stem.startswith(sep) else stem
            name = f"llm:{model_slug}[{variant}]"
            preds = load_llm_aligned(df, jp)
            n_present = int(np.sum(~np.isnan(preds)))
            if n_present == 0:
                continue
            llm_preds[name] = preds
            jsonl_paths.append(str(jp.resolve()))
            print(
                f"[repeated-splits] LLM aligned: {name}  rows_with_pred={n_present}/{len(df)}",
                flush=True,
            )
    return llm_preds, jsonl_paths


def cmd_repeated_splits(args: argparse.Namespace) -> None:
    """LR + RF + LLM over N repeated stratified 400/100 splits using saved best params.

    Reads RF ``best_params`` from ``metrics_sklearn.json`` (no nested grid search).
    LLM predictions are pre-aligned to the modeling table from each ``llm_*.jsonl``
    checkpoint, so each iteration just slices them with the same ``idx_te`` used
    for the tabular models.
    """
    import pandas as pd

    df = pd.read_csv(args.table, encoding="utf-8-sig")
    try:
        validate_modeling_table(df, need_text=False)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    feats = _feature_names(df)
    if not feats:
        raise SystemExit("No feature columns after config + tests CSV filter.")

    y_raw = pd.to_numeric(df["y_propa"], errors="coerce")
    df = df.loc[y_raw.notna()].reset_index(drop=True)
    y = y_raw.loc[y_raw.notna()].values.astype(float)
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if len(df) <= HOLDOUT_N:
        raise SystemExit(
            f"Need more than {HOLDOUT_N} rows for {HOLDOUT_N}-row holdout."
        )

    metrics_path = Path(args.metrics)
    if not metrics_path.is_file():
        raise SystemExit(
            f"Sklearn metrics JSON not found: {metrics_path}. "
            "Run `python run_model_benchmark.py sklearn` first."
        )
    sk_meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    rf_best = (sk_meta.get("rf_gridsearch") or {}).get("best_params")
    if not rf_best:
        raise SystemExit(
            f"Missing rf_gridsearch.best_params in {metrics_path}; rerun sklearn."
        )

    llm_preds, llm_paths = _collect_llm_aligned(df)
    if not llm_preds:
        print(
            "[repeated-splits] no LLM JSONL checkpoints found — running LR + RF only.",
            flush=True,
        )

    print(
        f"[repeated-splits] n_repeats={args.n_repeats}  base_seed={args.base_seed}  "
        f"holdout_n={HOLDOUT_N}  features={len(feats)}  "
        f"rf_best_params={rf_best}",
        flush=True,
    )

    strat = np.rint(y).clip(0, 4).astype(int)
    rows = repeated_splits(
        X,
        y,
        rf_best,
        llm_preds_by_name=llm_preds,
        n_repeats=int(args.n_repeats),
        holdout_n=HOLDOUT_N,
        base_seed=int(args.base_seed),
        strat=strat,
    )
    summary = {name: summarize_repeats(r) for name, r in rows.items()}

    for name, s in summary.items():
        print(
            f"  {name:<48s}  mse={s['mse_mean']:.4f} "
            f"95% CI=[{s['mse_ci_low']:.4f}, {s['mse_ci_high']:.4f}]  "
            f"r2={s['r2_mean']:.3f}  (n={s['n_repeats']})",
            flush=True,
        )

    payload = {
        "schema": "benchmark_repeated_splits_v1",
        "table": str(args.table.resolve()),
        "metrics_sklearn_json": str(metrics_path.resolve()),
        "llm_jsonl_paths": llm_paths,
        "n_repeats": int(args.n_repeats),
        "base_seed": int(args.base_seed),
        "holdout_n": int(HOLDOUT_N),
        "feature_selection_rule": feature_selection_rule(),
        "feature_count": len(feats),
        "features": feats,
        "rf_best_params": rf_best,
        "per_seed_rows": rows,
        "summary": summary,
    }
    write_metrics_json(METRICS_REPEATED_SPLITS_JSON, payload)


def cmd_bert_repeated_splits(args: argparse.Namespace) -> None:
    """Delegate to ``python -m modeling.bert_repeated_splits`` (heavy, GPU recommended)."""
    cmd = [
        sys.executable,
        "-m",
        "modeling.bert_repeated_splits",
        "--n-repeats",
        str(args.n_repeats),
        "--base-seed",
        str(args.base_seed),
    ]
    if getattr(args, "metrics", None) is not None:
        cmd += ["--metrics", str(args.metrics)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def cmd_compare(_args: argparse.Namespace) -> None:
    """Build ``data/model_comparison.csv`` from reliability + sklearn + BERT + LLM metrics JSONs."""
    import pandas as pd

    rows: list[dict] = []

    if METRICS_RELIABILITY_JSON.is_file():
        rel = json.loads(METRICS_RELIABILITY_JSON.read_text(encoding="utf-8"))
        nf = rel.get("noise_floor", {})
        n = rel.get("n")
        rows.append(
            _compare_row(
                "constant_mean (reliability)",
                "all_rows",
                n,
                {"mse": nf.get("mse_const_consensus"), "mae": nf.get("mae_const_consensus")},
                METRICS_RELIABILITY_JSON,
            )
        )
        rows.append(
            _compare_row(
                "single_annotator_vs_consensus",
                "all_rows",
                n,
                {
                    "mse": nf.get("mse_single_vs_consensus"),
                    "mae": nf.get("mae_single_vs_consensus"),
                },
                METRICS_RELIABILITY_JSON,
            )
        )

    if METRICS_SKLEARN_JSON.is_file():
        sk = json.loads(METRICS_SKLEARN_JSON.read_text(encoding="utf-8"))
        n_h = sk.get("n_holdout")
        for name, m in (sk.get("holdout") or {}).items():
            rows.append(
                _compare_row(f"sklearn:{name}", "holdout", n_h, m, METRICS_SKLEARN_JSON)
            )

    bert_path = BERT_CHECKPOINT_DIR / "metrics.json"
    if bert_path.is_file():
        be = json.loads(bert_path.read_text(encoding="utf-8"))
        rows.append(
            _compare_row(
                f"bert:{be.get('model_id', 'unknown')}",
                "holdout",
                be.get("holdout_n"),
                {
                    "mse": be.get("holdout_mse"),
                    "mae": be.get("holdout_mae"),
                    "rmse": be.get("holdout_rmse")
                    or ((be.get("holdout_mse") ** 0.5) if be.get("holdout_mse") else None),
                    "mse_ci_low": be.get("holdout_mse_ci_low"),
                    "mse_ci_high": be.get("holdout_mse_ci_high"),
                    "r2_vs_const": be.get("holdout_r2_vs_const"),
                    "r2_ci_low": be.get("holdout_r2_ci_low"),
                    "r2_ci_high": be.get("holdout_r2_ci_high"),
                },
                bert_path,
            )
        )

    for path in sorted(LLM_CHECKPOINT_DIR.glob("metrics_llm_*__*.json")):
        ll = json.loads(path.read_text(encoding="utf-8"))
        label = f"llm[{ll.get('variant')}]:{ll.get('model')}"
        reg_full = ll.get("regression_full_table") or ll.get("regression_all")
        n_full = ll.get("rows_scored_merged") or ll.get("rows_scored")
        if reg_full and n_full is not None:
            rows.append(
                _compare_row(label, "full_table", int(n_full), reg_full, path)
            )
        ho = ll.get("holdout") or {}
        if not ho:
            continue
        rows.append(
            _compare_row(label, "holdout", ho.get("n"), ho, path)
        )

    def _rmse_from_summary(s: dict) -> float | None:
        """Prefer the native ``rmse_mean`` (mean of per-seed RMSEs); fall back to
        ``sqrt(mse_mean)`` for legacy summaries that predate native RMSE."""
        rmse_mean = s.get("rmse_mean")
        if isinstance(rmse_mean, (int, float)) and rmse_mean == rmse_mean:
            return float(rmse_mean)
        mse_mean = s.get("mse_mean")
        if isinstance(mse_mean, (int, float)) and mse_mean == mse_mean and mse_mean >= 0:
            return float(mse_mean) ** 0.5
        return None

    if METRICS_REPEATED_SPLITS_JSON.is_file():
        rs = json.loads(METRICS_REPEATED_SPLITS_JSON.read_text(encoding="utf-8"))
        n_rep = rs.get("n_repeats")
        eval_setup = f"repeated_splits_{n_rep}" if n_rep else "repeated_splits"
        for name, s in (rs.get("summary") or {}).items():
            if name in ("lr", "rf"):
                label = f"sklearn:{name}"
            else:
                label = name
            rows.append(
                _compare_row(
                    label,
                    eval_setup,
                    s.get("n_repeats"),
                    {
                        "mse": s.get("mse_mean"),
                        "mae": s.get("mae_mean"),
                        "rmse": _rmse_from_summary(s),
                        "mse_ci_low": s.get("mse_ci_low"),
                        "mse_ci_high": s.get("mse_ci_high"),
                        "r2_vs_const": s.get("r2_mean"),
                        "r2_ci_low": s.get("r2_ci_low"),
                        "r2_ci_high": s.get("r2_ci_high"),
                    },
                    METRICS_REPEATED_SPLITS_JSON,
                )
            )

    bert_rs_path = BERT_CHECKPOINT_DIR / "metrics_repeated_splits.json"
    if bert_rs_path.is_file():
        be_rs = json.loads(bert_rs_path.read_text(encoding="utf-8"))
        s = be_rs.get("summary") or {}
        n_rep = be_rs.get("n_repeats")
        eval_setup = f"repeated_splits_{n_rep}" if n_rep else "repeated_splits"
        rows.append(
            _compare_row(
                f"bert:{be_rs.get('model_id', 'unknown')}",
                eval_setup,
                s.get("n_repeats"),
                {
                    "mse": s.get("mse_mean"),
                    "mae": s.get("mae_mean"),
                    "rmse": _rmse_from_summary(s),
                    "mse_ci_low": s.get("mse_ci_low"),
                    "mse_ci_high": s.get("mse_ci_high"),
                    "r2_vs_const": s.get("r2_mean"),
                    "r2_ci_low": s.get("r2_ci_low"),
                    "r2_ci_high": s.get("r2_ci_high"),
                },
                bert_rs_path,
            )
        )

    if not rows:
        print("No metrics files found to compare.")
        return

    df_out = pd.DataFrame(rows, columns=list(_COMPARE_COLUMNS))
    df_out = df_out.sort_values(by=["eval_setup", "mse"], na_position="last").reset_index(drop=True)
    MODEL_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(MODEL_COMPARISON_CSV, index=False, encoding="utf-8-sig")
    print(df_out.to_string(index=False))
    print(f"\nWrote comparison: {MODEL_COMPARISON_CSV}")


def cmd_all(args: argparse.Namespace) -> None:
    """Run export-table, reliability, sklearn, both LLM variants, tyyp, then BERT (see subcommands)."""
    llm_no = argparse.Namespace(table=args.table, variant="no_features")
    llm_with = argparse.Namespace(table=args.table, variant="with_features")

    steps: list[tuple[str, callable]] = []
    if not args.skip_export:
        steps.append(("export-table", lambda a=args: cmd_export_table(a)))
    steps += [
        ("reliability", lambda a=args: cmd_reliability(a)),
        ("sklearn", lambda a=args: cmd_sklearn(a)),
        ("llm (no_features)", lambda a=llm_no: cmd_llm(a)),
        ("llm (with_features)", lambda a=llm_with: cmd_llm(a)),
        ("tyyp", lambda a=args: cmd_tyyp(a)),
        ("bert", lambda a=args: cmd_bert(a)),
    ]
    for i, (name, fn) in enumerate(steps, start=1):
        print(f"\n>>> Pipeline step {i}/{len(steps)}: {name}\n", flush=True)
        fn()
    print(f"\n>>> Pipeline complete ({len(steps)}/{len(steps)} steps).\n", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Propaganda modeling benchmark (config-driven features)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("reliability", help="Cronbach α + Cohen κ (Uku vs Simon)")
    pr.add_argument("--uku", type=Path, default=UKU_LABELING_CSV)
    pr.add_argument("--simon", type=Path, default=SIMON_LABELING_CSV)
    pr.set_defaults(func=cmd_reliability)

    pe = sub.add_parser("export-table", help="Merge Uku+Simon+features → modeling CSV (path in config)")
    pe.add_argument("--uku", type=Path, default=UKU_LABELING_CSV)
    pe.add_argument("--simon", type=Path, default=SIMON_LABELING_CSV)
    pe.add_argument("--features", type=Path, default=LABELING_WITH_FEATURES_CSV)
    pe.set_defaults(func=cmd_export_table)

    ps = sub.add_parser(
        "sklearn",
        help="OLS on train→holdout; RF via GridSearchCV (inner 5-fold on train)→holdout; writes manifest + predictions.",
    )
    ps.add_argument("--table", type=Path, default=MODELING_TABLE_CSV)
    ps.set_defaults(func=cmd_sklearn)

    pi = sub.add_parser(
        "sklearn-importances",
        help="Refit LR+RF on train (no grid); write coef + RF importances. Compares holdout MSE to metrics JSON (read-only).",
    )
    pi.add_argument("--table", type=Path, default=MODELING_TABLE_CSV)
    pi.add_argument(
        "--metrics",
        type=Path,
        default=METRICS_SKLEARN_JSON,
        help="benchmark_sklearn_v5 JSON (must list features + rf_gridsearch.best_params).",
    )
    pi.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON only (default: sklearn_feature_importances.json beside --metrics). Does not overwrite --metrics.",
    )
    pi.add_argument(
        "--sanity-mse-atol",
        type=float,
        default=1e-10,
        help="Max allowed |MSE_refit − MSE_reported| per model for sanity check (default: 1e-10).",
    )
    pi.add_argument(
        "--allow-sanity-fail",
        action="store_true",
        help="If sanity check fails, still write --out and exit 0 (default: exit 1).",
    )
    pi.set_defaults(func=cmd_sklearn_importances)

    pl = sub.add_parser(
        "llm",
        help="OpenAI Chat Completions + structured output; JSONL checkpoint",
    )
    pl.add_argument("--table", type=Path, default=MODELING_TABLE_CSV)
    pl.add_argument(
        "--variant",
        choices=("no_features", "with_features"),
        required=True,
        help="no_features = text only; with_features = same + numeric features as JSON before the post.",
    )
    pl.add_argument(
        "--model",
        default=None,
        help="OpenAI model id (overrides OPENAI_MODEL / config.LLM_MODEL).",
    )
    pl.set_defaults(func=cmd_llm)

    pt = sub.add_parser("tyyp", help="Propaganda type (Tüüp): TF–IDF + RF on post text")
    pt.add_argument("--table", type=Path, default=MODELING_TABLE_CSV)
    pt.set_defaults(func=cmd_tyyp)

    sub.add_parser(
        "bert",
        help="Fine-tune one sequence model for 0–4 regression (see config.BERT_MODEL_ID; needs torch, transformers, datasets, accelerate).",
    ).set_defaults(func=cmd_bert)

    prs = sub.add_parser(
        "repeated-splits",
        help="LR + RF + LLM over N stratified 400/100 splits with saved best params; writes summary CIs.",
    )
    prs.add_argument("--table", type=Path, default=MODELING_TABLE_CSV)
    prs.add_argument(
        "--metrics",
        type=Path,
        default=METRICS_SKLEARN_JSON,
        help="Source of RF best_params (default: metrics_sklearn.json).",
    )
    prs.add_argument("--n-repeats", type=int, default=500)
    prs.add_argument(
        "--base-seed",
        type=int,
        default=100,
        help="Seed schedule = range(base_seed, base_seed+n_repeats). "
        "Default 100 keeps the fixed-holdout seed 42 untouched and aligns "
        "with modeling.bert_repeated_splits.",
    )
    prs.set_defaults(func=cmd_repeated_splits)

    pbrs = sub.add_parser(
        "bert-repeated-splits",
        help="BERT over N stratified 400/100 splits with saved final_* hparams "
        "(GPU strongly recommended; hours).",
    )
    pbrs.add_argument("--n-repeats", type=int, default=500)
    pbrs.add_argument("--base-seed", type=int, default=100)
    pbrs.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help=(
            "BERT metrics JSON with final_* hparams. Defaults to "
            "checkpoints/bert_propa_full_run/metrics.json "
            "(falls back to checkpoints/bert_propa/metrics.json)."
        ),
    )
    pbrs.set_defaults(func=cmd_bert_repeated_splits)

    sub.add_parser(
        "compare",
        help="Aggregate metrics JSONs into data/model_comparison.csv (LLM: full_table + holdout; sklearn/BERT: holdout).",
    ).set_defaults(func=cmd_compare)

    pa = sub.add_parser(
        "all",
        help="Full run: export-table, reliability, sklearn, llm×2, tyyp, bert (same paths as subcommands).",
    )
    pa.add_argument("--uku", type=Path, default=UKU_LABELING_CSV)
    pa.add_argument("--simon", type=Path, default=SIMON_LABELING_CSV)
    pa.add_argument("--features", type=Path, default=LABELING_WITH_FEATURES_CSV)
    pa.add_argument("--table", type=Path, default=MODELING_TABLE_CSV)
    pa.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export-table (use when modeling_table.csv is already up to date).",
    )
    pa.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
