"""Inter-rater reliability: Cronbach's alpha (2 raters = internal consistency of the scale)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def cronbach_alpha(items: pd.DataFrame) -> float:
    """
    Cronbach's α for rows-as-subjects, columns-as-items (e.g. two raters).

    items: shape (n, k), numeric. Returns nan if undefined.
    """
    items = items.astype(float)
    items = items.dropna(how="any")
    n, k = items.shape
    if n < 2 or k < 2:
        return float("nan")
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var <= 0 or np.isnan(total_var):
        return float("nan")
    return float((k / (k - 1)) * (1.0 - item_vars.sum() / total_var))


def agreement_report(
    propa_uku: pd.Series,
    propa_simon: pd.Series,
) -> dict:
    """Cronbach α, quadratic weighted κ, accuracy, Pearson r, plus a noise floor.

    The noise floor describes the upper bound on what any model can achieve
    against the consensus mean target. It includes:

    - ``var_y_consensus``: Var of the consensus mean — equal to MSE of
      "predict the mean" on the same set, i.e. an absolute floor for any model
      worse than the constant baseline.
    - ``mae_const_consensus``: MAE of "predict the mean" baseline.
    - ``mse_single_vs_consensus`` / ``mae_single_vs_consensus``: average squared
      / absolute error of *each individual coder* against the consensus mean.
      A model whose MSE matches this is performing as well as a single human
      coder relative to the agreed-upon truth.
    """
    u = pd.to_numeric(propa_uku, errors="coerce")
    s = pd.to_numeric(propa_simon, errors="coerce")
    mask = u.notna() & s.notna()
    u = u[mask].astype(float)
    s = s[mask].astype(float)

    out: dict = {"n": int(len(u))}
    if len(u) == 0:
        return out

    u_int = u.astype(int)
    s_int = s.astype(int)
    mat = pd.DataFrame({"uku": u_int.values, "simon": s_int.values})
    out["cronbach_alpha"] = cronbach_alpha(mat)

    try:
        out["cohen_kappa_quadratic"] = float(
            cohen_kappa_score(u_int.values, s_int.values, weights="quadratic")
        )
    except Exception:
        out["cohen_kappa_quadratic"] = float("nan")

    try:
        out["cohen_kappa_linear"] = float(
            cohen_kappa_score(u_int.values, s_int.values, weights="linear")
        )
    except Exception:
        out["cohen_kappa_linear"] = float("nan")

    out["accuracy_exact"] = float((u_int.values == s_int.values).mean())
    out["pearson_r"] = float(mat["uku"].corr(mat["simon"]))

    consensus = (u.values + s.values) / 2.0
    diffs_u = u.values - consensus
    diffs_s = s.values - consensus
    var_y = float(np.var(consensus))
    out["noise_floor"] = {
        "var_y_consensus": var_y,
        "mse_const_consensus": var_y,
        "mae_const_consensus": float(np.mean(np.abs(consensus - consensus.mean()))),
        "mse_single_vs_consensus": float(
            np.mean(np.concatenate([diffs_u ** 2, diffs_s ** 2]))
        ),
        "mae_single_vs_consensus": float(
            np.mean(np.concatenate([np.abs(diffs_u), np.abs(diffs_s)]))
        ),
    }

    return out
