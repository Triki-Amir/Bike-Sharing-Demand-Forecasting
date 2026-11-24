"""Zero-inflated Poisson regression on weekend, temperature and rain (statsmodels)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

FEATURES = ["is_weekend", "temp_mean", "precip_mm"]
FALLBACK_WEEKS = 8


def _fallback(y: np.ndarray, n: int) -> np.ndarray:
    """Mean of the last 8 weeks, repeated n times."""
    recent = y[-7 * FALLBACK_WEEKS :]
    return np.full(n, float(recent.mean()) if len(recent) else 0.0)


def zip_fit_predict_info(train: pd.DataFrame, future: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Fit a ZIP model on train and predict future; returns (forecast, used_fallback)."""
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    y = train["y"].to_numpy(dtype=float)
    fallback = _fallback(y, len(future))
    if (y > 0).sum() < 10 or len(future) == 0:
        return fallback, True
    X = sm.add_constant(train[FEATURES].astype(float), has_constant="add")
    Xf = sm.add_constant(future[FEATURES].astype(float), has_constant="add")
    if X.isna().any().any() or Xf.isna().any().any():
        return fallback, True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ZeroInflatedPoisson(y, X, inflation="logit").fit(disp=0, maxiter=200)
            pred = np.asarray(
                res.predict(Xf, exog_infl=np.ones((len(Xf), 1)), which="mean"), dtype=float
            )
    except Exception:  # noqa: BLE001 - any numerical failure falls back
        return fallback, True
    # Guard against exploding coefficients: keep predictions in a plausible range.
    if not np.all(np.isfinite(pred)) or pred.max() > 10 * y.max() + 10:
        return fallback, True
    return np.clip(pred, 0, None), False


def zip_fit_predict_flag(train: pd.DataFrame, future: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Backtest-compatible fit_predict returning (forecast, fallback flag per day)."""
    pred, used = zip_fit_predict_info(train, future)
    return pred, np.full(len(pred), float(used))


def zip_fit_predict(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Backtest-compatible fit_predict for the ZIP model (point forecast only)."""
    return zip_fit_predict_info(train, future)[0]
