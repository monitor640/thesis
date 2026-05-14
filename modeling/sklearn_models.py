"""Tabular regression helpers for the propaganda 0–4 consensus target.

``make_pipeline`` + ``fit_predict`` are used by the benchmark (OLS + imputer + scaler).
``cv_regress`` is kept for ad-hoc / notebook diagnostics only — benchmark metrics use
holdout evaluation and RF inner GridSearchCV instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.metrics import evaluate_regression


def make_pipeline(estimator: BaseEstimator, *, scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("est", estimator))
    return Pipeline(steps)


def cv_regress(
    X: np.ndarray,
    y: np.ndarray,
    estimator: BaseEstimator,
    *,
    scale: bool,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """K-fold out-of-fold predictions + regression metrics (optional debugging; not used in ``run_model_benchmark``)."""
    pipe = make_pipeline(estimator, scale=scale)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipe, X, y, cv=kf)
    return y_pred, evaluate_regression(y, y_pred)


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    estimator: BaseEstimator,
    *,
    scale: bool,
) -> np.ndarray:
    pipe = make_pipeline(estimator, scale=scale).fit(X_train, y_train)
    return pipe.predict(X_test)
