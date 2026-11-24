"""Rolling-origin backtest, tuning and scoring shared by the notebooks and the local runner."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bikecast import config
from bikecast import metrics as M
from bikecast.markov import make_markov_fit_predict

KEYS = ["unique_id", "ds", "cutoff"]
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], "np.ndarray | tuple[np.ndarray, np.ndarray]"]


# --------------------------------------------------------------------------- cutoffs
def rolling_cutoffs(
    last_date: str | pd.Timestamp, h: int = config.HORIZON, n_windows: int = config.N_WINDOWS
) -> list[pd.Timestamp]:
    """Cutoffs of n_windows non-overlapping h-day windows that end on last_date."""
    last = pd.Timestamp(last_date)
    return [last - pd.Timedelta(days=h * (n_windows - i)) for i in range(n_windows)]


# --------------------------------------------------------------------------- generic loop
def _backtest_station(
    args: tuple[str, pd.DataFrame, FitPredict, str, list[pd.Timestamp], int, str],
) -> list[pd.DataFrame]:
    """Run fit_predict for one station on every cutoff."""
    sid, g, fit_predict, name, cutoffs, h, extra = args
    out = []
    for cutoff in cutoffs:
        train = g[g.ds <= cutoff]
        future = g[(g.ds > cutoff) & (g.ds <= cutoff + pd.Timedelta(days=h))]
        if future.empty:
            continue
        res = fit_predict(train, future)
        frame = {"unique_id": sid, "ds": future.ds.values, "cutoff": cutoff}
        if isinstance(res, tuple):
            frame[name] = np.asarray(res[0], dtype=float)
            frame[f"{name}_{extra}"] = np.asarray(res[1], dtype=float)
        else:
            frame[name] = np.asarray(res, dtype=float)
        out.append(pd.DataFrame(frame))
    return out


def backtest(
    df: pd.DataFrame,
    fit_predict: FitPredict,
    name: str,
    cutoffs: Sequence,
    h: int = config.HORIZON,
    n_jobs: int = 1,
    extra_name: str = "p_sale",
) -> pd.DataFrame:
    """Run any fit_predict(train_df, future_df) on the same cutoffs for every station.

    fit_predict may return an array of forecasts or a tuple (forecasts, extra); in the latter
    case an extra column ``<name>_<extra_name>`` is produced (e.g. ``Markov_p_sale``).
    """
    cutoffs = [pd.Timestamp(c) for c in cutoffs]
    tasks = [
        (sid, g, fit_predict, name, cutoffs, h, extra_name)
        for sid, g in df.groupby("unique_id", sort=True)
    ]
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            parts = list(ex.map(_backtest_station, tasks, chunksize=max(1, len(tasks) // (4 * n_jobs))))
    else:
        parts = [_backtest_station(t) for t in tasks]
    frames = [f for p in parts for f in p]
    if not frames:
        return pd.DataFrame(columns=[*KEYS, name])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- statsforecast
def statsforecast_cv(
    df: pd.DataFrame,
    tsb_alphas: tuple[float, float] = config.DEFAULT_TSB_ALPHAS,
    h: int = config.HORIZON,
    n_windows: int = config.N_WINDOWS,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Seasonal Naive, Croston and TSB with statsforecast rolling cross-validation."""
    from statsforecast import StatsForecast
    from statsforecast.models import TSB, CrostonClassic, SeasonalNaive

    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=config.SEASON_LENGTH),
            CrostonClassic(),
            TSB(alpha_d=tsb_alphas[0], alpha_p=tsb_alphas[1]),
        ],
        freq="D",
        n_jobs=n_jobs,
    )
    cv = sf.cross_validation(df=df[["unique_id", "ds", "y"]], h=h, step_size=h, n_windows=n_windows)
    return _normalise_cv(cv)


def _normalise_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Make statsforecast output a flat frame with typed keys."""
    if "unique_id" not in cv.columns:
        cv = cv.reset_index()
    cv["ds"] = pd.to_datetime(cv["ds"])
    cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    cv["unique_id"] = cv["unique_id"].astype(str)
    return cv


def _truncate_for_cutoffs(df: pd.DataFrame, cutoffs: Sequence, h: int) -> pd.DataFrame:
    """Keep data up to the end of the last tuning window (so test days stay unseen)."""
    end = pd.Timestamp(max(cutoffs)) + pd.Timedelta(days=h)
    return df[df.ds <= end]


def tune_tsb(
    df: pd.DataFrame,
    tuning_cutoffs: Sequence,
    grid: Sequence[float] = config.TSB_ALPHA_GRID,
    h: int = config.HORIZON,
    n_jobs: int = 1,
) -> tuple[tuple[float, float], pd.DataFrame]:
    """Pick (alpha_d, alpha_p) with the lowest MAE on the tuning cutoffs only."""
    from statsforecast import StatsForecast
    from statsforecast.models import TSB

    pairs = [(a, b) for a in grid for b in grid]
    models = [TSB(alpha_d=a, alpha_p=b, alias=f"TSB_{a}_{b}") for a, b in pairs]
    data = _truncate_for_cutoffs(df, tuning_cutoffs, h)
    sf = StatsForecast(models=models, freq="D", n_jobs=n_jobs)
    cv = _normalise_cv(
        sf.cross_validation(df=data[["unique_id", "ds", "y"]], h=h, step_size=h,
                            n_windows=len(tuning_cutoffs))
    )
    scores = pd.DataFrame(
        [{"alpha_d": a, "alpha_p": b, "mae": M.mae(cv["y"], cv[f"TSB_{a}_{b}"])} for a, b in pairs]
    ).sort_values("mae", kind="stable")
    best = scores.iloc[0]
    return (float(best.alpha_d), float(best.alpha_p)), scores.reset_index(drop=True)


def tune_markov_window(
    df: pd.DataFrame,
    tuning_cutoffs: Sequence,
    grid: Sequence[int] = config.MARKOV_WINDOW_GRID,
    h: int = config.HORIZON,
) -> tuple[int, pd.DataFrame]:
    """Pick the Markov training window with the lowest MAE on the tuning cutoffs only."""
    data = _truncate_for_cutoffs(df, tuning_cutoffs, h)
    rows = []
    for w in grid:
        fc = backtest(data, make_markov_fit_predict(w), "Markov", tuning_cutoffs, h=h)
        fc = fc.merge(data[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
        rows.append({"window": w, "mae": M.mae(fc["y"], fc["Markov"])})
    scores = pd.DataFrame(rows).sort_values("mae", kind="stable").reset_index(drop=True)
    return int(scores.iloc[0].window), scores


# --------------------------------------------------------------------------- scoring
def combine(cv: pd.DataFrame, *others: pd.DataFrame) -> pd.DataFrame:
    """Merge the statsforecast CV frame with the custom model frames on (station, day, cutoff)."""
    fc = cv.copy()
    for o in others:
        o = o.copy()
        o["unique_id"] = o["unique_id"].astype(str)
        o["ds"] = pd.to_datetime(o["ds"])
        o["cutoff"] = pd.to_datetime(o["cutoff"])
        fc = fc.merge(o, on=KEYS, how="inner")
    return fc.sort_values(KEYS).reset_index(drop=True)


def score(
    fc: pd.DataFrame,
    df: pd.DataFrame,
    models: Sequence[str] = config.MODELS,
    train_end: str | pd.Timestamp = config.TRAIN_END,
) -> pd.DataFrame:
    """Per station x model: MAE, bias and MASE over all test days (MASE scale from train)."""
    seg = df.drop_duplicates("unique_id").set_index("unique_id")["segment"]
    train_end = pd.Timestamp(train_end)
    train = df[df.ds <= train_end]
    y_trains = {sid: g["y"].to_numpy() for sid, g in train.groupby("unique_id")}
    rows = []
    for sid, g in fc.groupby("unique_id"):
        y_train = y_trains.get(sid, np.array([]))
        for m in models:
            row = {
                "station_id": sid,
                "segment": seg.get(sid, "unknown"),
                "model": m,
                "mae": M.mae(g.y, g[m]),
                "bias": M.bias_pct(g.y, g[m]),
                "mase": M.mase(g.y, g[m], y_train),
                "zero_day_accuracy": np.nan,
            }
            if f"{m}_p_sale" in g.columns:
                row["zero_day_accuracy"] = M.zero_day_accuracy(g.y, g[f"{m}_p_sale"])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """Segment x model summary: median MASE, mean MAE, mean bias and station count."""
    return (
        res.groupby(["segment", "model"])
        .agg(
            stations=("station_id", "nunique"),
            mase_median=("mase", "median"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
        )
        .reset_index()
    )


def mase_table(summary: pd.DataFrame, models: Sequence[str] = config.MODELS) -> pd.DataFrame:
    """Pivot to rows = segments (+ 'all'), columns = stations + models, values = median MASE."""
    order = [s for s in config.SEGMENTS if s in set(summary.segment)]
    piv = summary.pivot(index="segment", columns="model", values="mase_median")
    piv = piv.reindex(index=order, columns=[m for m in models if m in piv.columns])
    counts = summary.groupby("segment")["stations"].max().reindex(order)
    piv.insert(0, "stations", counts.astype(int))
    return piv


def to_long(fc: pd.DataFrame, models: Sequence[str] = config.MODELS) -> pd.DataFrame:
    """Long forecasts table: station_id, date, cutoff, model, y, y_hat, p_sale."""
    parts = []
    for m in models:
        part = fc[["unique_id", "ds", "cutoff", "y"]].copy()
        part["model"] = m
        part["y_hat"] = fc[m].astype(float)
        part["p_sale"] = fc[f"{m}_p_sale"].astype(float) if f"{m}_p_sale" in fc.columns else np.nan
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={"unique_id": "station_id", "ds": "date"})


def headline(summary: pd.DataFrame, segments: Sequence[str] = ("intermittent", "lumpy")) -> dict:
    """Best model and MASE reduction vs Seasonal Naive for the given segments."""
    out = {}
    for s in segments:
        sub = summary[summary.segment == s].set_index("model")["mase_median"]
        if sub.empty or "SeasonalNaive" not in sub.index:
            continue
        best = sub.drop("SeasonalNaive").idxmin()
        base = sub["SeasonalNaive"]
        out[s] = {
            "best_model": best,
            "best_mase": float(sub[best]),
            "naive_mase": float(base),
            "reduction_pct": float(100 * (base - sub[best]) / base) if base > 0 else float("nan"),
        }
    return out
