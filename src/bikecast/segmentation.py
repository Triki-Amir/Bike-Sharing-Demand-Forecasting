"""Demand segmentation with ADI and CV² (Syntetos & Boylan, 2005)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bikecast import config


def adi_cv2(y: np.ndarray) -> tuple[float, float]:
    """Return (ADI, CV²) of a daily series: days per active day, squared CV of non-zero sizes."""
    y = np.asarray(y, dtype=float)
    nz = y[y > 0]
    if len(nz) < 2:
        return float("inf"), float("nan")
    return len(y) / len(nz), float((nz.std(ddof=1) / nz.mean()) ** 2)


def classify(
    adi: float, cv2: float, adi_cut: float = config.ADI_CUT, cv2_cut: float = config.CV2_CUT
) -> str:
    """Map (ADI, CV²) to smooth / erratic / intermittent / lumpy."""
    if adi < adi_cut:
        return "smooth" if cv2 < cv2_cut else "erratic"
    return "intermittent" if cv2 < cv2_cut else "lumpy"


def segment_stations(
    station_daily: pd.DataFrame,
    train_end: str = config.TRAIN_END,
    min_nonzero: int = config.MIN_NONZERO_DAYS,
) -> pd.DataFrame:
    """Compute per-station ADI, CV² and segment on the training period (mirrors the SQL)."""
    train = station_daily[pd.to_datetime(station_daily["date"]) <= pd.Timestamp(train_end)]
    rows = []
    for sid, g in train.groupby("station_id", sort=True):
        y = g["trips"].to_numpy(dtype=float)
        nz = y[y > 0]
        n_days, n_nonzero = len(y), len(nz)
        if n_nonzero < min_nonzero:
            continue
        adi, cv2 = adi_cv2(y)
        rows.append(
            {
                "station_id": sid,
                "n_days": n_days,
                "n_nonzero": n_nonzero,
                "mean_daily_trips": float(y.mean()),
                "pct_zero_days": 1 - n_nonzero / n_days,
                "adi": adi,
                "cv2": cv2,
                "segment": classify(adi, cv2),
            }
        )
    cols = ["station_id", "n_days", "n_nonzero", "mean_daily_trips", "pct_zero_days", "adi",
            "cv2", "segment"]
    return pd.DataFrame(rows, columns=cols)
