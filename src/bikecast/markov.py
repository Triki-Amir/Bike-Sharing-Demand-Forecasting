"""Two-state Markov chain (trip / no trip) combined with a Negative Binomial size model.

Forecast for day k = P(trip on day k | today's state) x expected size on an active day.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd

from bikecast import config


def transition_matrix(y: np.ndarray) -> np.ndarray:
    """Estimate the 2x2 active/inactive transition matrix with Laplace (+1) smoothing."""
    s = (np.asarray(y) > 0).astype(int)
    prev, nxt = s[:-1], s[1:]
    n00 = np.sum((prev == 0) & (nxt == 0))
    n01 = np.sum((prev == 0) & (nxt == 1))
    n10 = np.sum((prev == 1) & (nxt == 0))
    n11 = np.sum((prev == 1) & (nxt == 1))
    p01 = (n01 + 1) / (n00 + n01 + 2)
    p11 = (n11 + 1) / (n10 + n11 + 2)
    return np.array([[1 - p01, p01], [1 - p11, p11]])


def size_params(y: np.ndarray) -> dict:
    """Fit the size of active days, (trips - 1), by method of moments: NegBin or Poisson."""
    y = np.asarray(y, dtype=float)
    z = y[y > 0] - 1  # shift so the size starts at 0
    m = float(z.mean()) if len(z) else 0.0
    v = float(z.var(ddof=1)) if len(z) > 1 else m
    if v > m > 0:  # over-dispersed -> Negative Binomial
        r = m**2 / (v - m)
        return {"dist": "negbin", "r": r, "p": r / (r + m), "mean_size": 1 + m}
    return {"dist": "poisson", "lam": m, "mean_size": 1 + m}


def markov_forecast(
    y: np.ndarray, h: int = config.HORIZON, window: int = config.DEFAULT_MARKOV_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Return (expected trips, probability of at least one trip) for the next h days."""
    y = np.asarray(y, dtype=float)[-window:]
    if len(y) == 0:
        return np.zeros(h), np.zeros(h)
    P = transition_matrix(y)
    size = size_params(y)
    state = np.array([0.0, 1.0]) if y[-1] > 0 else np.array([1.0, 0.0])
    p_sale = np.empty(h)
    for k in range(h):
        state = state @ P
        p_sale[k] = state[1]
    return p_sale * size["mean_size"], p_sale


def simulate_paths(
    y: np.ndarray,
    h: int = config.HORIZON,
    window: int = config.DEFAULT_MARKOV_WINDOW,
    n_paths: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Simulate (n_paths, h) future trip counts from the fitted chain and size distribution."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)[-window:]
    P = transition_matrix(y)
    size = size_params(y)
    state = np.full(n_paths, int(y[-1] > 0)) if len(y) else np.zeros(n_paths, dtype=int)
    out = np.zeros((n_paths, h))
    for k in range(h):
        p_active = P[state, 1]
        state = (rng.random(n_paths) < p_active).astype(int)
        if size["dist"] == "negbin":
            sizes = 1 + rng.negative_binomial(size["r"], size["p"], n_paths)
        else:
            sizes = 1 + rng.poisson(size["lam"], n_paths)
        out[:, k] = state * sizes
    return out


def prediction_interval(
    y: np.ndarray, h: int = config.HORIZON, q: tuple[float, float] = (0.1, 0.9), **kw
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower and upper simulated quantiles for each of the next h days."""
    paths = simulate_paths(y, h=h, **kw)
    return np.quantile(paths, q[0], axis=0), np.quantile(paths, q[1], axis=0)


def markov_fit_predict_p(
    train: pd.DataFrame, future: pd.DataFrame, window: int = config.DEFAULT_MARKOV_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Backtest-compatible fit_predict returning (y_hat, p_sale)."""
    return markov_forecast(train["y"].to_numpy(), h=len(future), window=window)


def make_markov_fit_predict(
    window: int = config.DEFAULT_MARKOV_WINDOW,
) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]]:
    """Build a picklable fit_predict with a fixed window that also returns p_sale."""
    return partial(markov_fit_predict_p, window=window)


def markov_fit_predict(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Backtest-compatible fit_predict with the default window (point forecast only)."""
    y_hat, _ = markov_forecast(train["y"].to_numpy(), h=len(future))
    return y_hat
