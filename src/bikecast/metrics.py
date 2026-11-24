"""Forecast error metrics that work with zero-heavy series."""

from __future__ import annotations

import numpy as np

from bikecast import config


def mae(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute error, in trips."""
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(f, dtype=float))))


def bias_pct(y: np.ndarray, f: np.ndarray) -> float:
    """Relative bias, sum-based so it works with zeros. Positive = over-forecast."""
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    return float((f.sum() - y.sum()) / max(y.sum(), 1e-9))


def mase(y: np.ndarray, f: np.ndarray, y_train: np.ndarray, m: int = config.SEASON_LENGTH) -> float:
    """MAE scaled by the in-sample MAE of the seasonal naive forecast (lag m)."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return float("nan")
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(mae(y, f) / scale) if scale > 0 else float("nan")


def zero_day_accuracy(y: np.ndarray, p_sale: np.ndarray, threshold: float = 0.5) -> float:
    """Share of days where 'trip or no trip' was predicted correctly."""
    return float(np.mean((np.asarray(p_sale) >= threshold) == (np.asarray(y) > 0)))
