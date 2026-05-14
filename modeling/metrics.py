from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error


def write_metrics_json(path: Path, payload: dict) -> None:
    """Write a JSON-serializable metrics dict (numpy scalars / arrays converted)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(o: object) -> object:
        if isinstance(o, dict):
            return {str(k): _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    path.write_text(json.dumps(_conv(payload), indent=2), encoding="utf-8")
    print(f"Wrote metrics: {path}")


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    """Percentile bootstrap CIs (95%) for MSE, MAE and R² on paired (y_true, y_pred).

    R² is computed per resample as ``1 - mse_i / var(y_true_i)``, consistent with the
    reported ``r2_vs_const``. Resamples with zero variance (all-identical labels) are
    dropped from the R² quantile via ``nanquantile``.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 2 or n_resamples <= 0:
        return {}
    mse = np.empty(n_resamples, dtype=float)
    mae = np.empty(n_resamples, dtype=float)
    r2 = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt_i = y_true[idx]
        diff = yt_i - y_pred[idx]
        mse[i] = float(np.mean(diff * diff))
        mae[i] = float(np.mean(np.abs(diff)))
        var_i = float(np.var(yt_i))
        r2[i] = (1.0 - mse[i] / var_i) if var_i > 0 else np.nan
    return {
        "mse_ci_low": float(np.quantile(mse, 0.025)),
        "mse_ci_high": float(np.quantile(mse, 0.975)),
        "mae_ci_low": float(np.quantile(mae, 0.025)),
        "mae_ci_high": float(np.quantile(mae, 0.975)),
        "r2_ci_low": float(np.nanquantile(r2, 0.025)),
        "r2_ci_high": float(np.nanquantile(r2, 0.975)),
        "ci_n_resamples": int(n_resamples),
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Regression metrics + bootstrap 95% CIs + R² vs the test-set constant mean.

    R² vs constant uses the test-set mean as the baseline predictor:
    ``r2_vs_const = 1 - MSE / Var(y_true)``. Negative values mean the model is
    worse than predicting the mean (common on small holdouts).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    # Compare rounded to nearest int (ordinal scale) for a discrete-style accuracy.
    acc_round = float(
        accuracy_score(
            np.rint(y_true).clip(0, 4).astype(int),
            np.rint(y_pred).clip(0, 4).astype(int),
        )
    )
    var_y = float(np.var(y_true))
    r2_const = float(1.0 - mse / var_y) if var_y > 0 else float("nan")
    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "accuracy_rounded_0_4": acc_round,
        "var_y_true": var_y,
        "r2_vs_const": r2_const,
        "report": (
            f"MSE={mse:.4f} MAE={mae:.4f} RMSE={rmse:.4f}  "
            f"acc(rint)={acc_round:.4f}  R²(const)={r2_const:.3f}"
        ),
    }
    out.update(_bootstrap_ci(y_true, y_pred, n_resamples=bootstrap, seed=bootstrap_seed))
    return out


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }
