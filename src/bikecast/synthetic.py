"""Synthetic Capital-Bikeshare-like data for CI and tests (same file formats as the real data).

Generates monthly trip CSVs with stations of known demand types (smooth, erratic, intermittent,
lumpy), plus dirty rows that silver must remove: duplicates, dockless trips without a station
and trips that are too short. One station opens in March 2026 and must be excluded from gold.
"""

from __future__ import annotations

import pathlib

import holidays as hol
import numpy as np
import pandas as pd

from bikecast import config
from bikecast.pipeline import TRIP_COLUMNS

# station_id -> (kind, parameter set)
PROFILES = {
    "smooth": {"p_active": 1.0, "mean": 18.0, "disp": None},
    "erratic": {"p_active": 0.97, "mean": 6.0, "disp": 0.8},
    "intermittent": {"p_active": 0.30, "mean": 1.2, "disp": None},
    "lumpy": {"p_active": 0.30, "mean": 5.0, "disp": 0.5},
}


def _daily_counts(kind: str, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Daily trip counts for one station of a given demand type."""
    p = PROFILES[kind]
    active = rng.random(n_days) < p["p_active"]
    if p["disp"] is None:
        size = rng.poisson(p["mean"], n_days)
    else:  # Negative Binomial with mean `mean` and shape `disp` (lower = more erratic)
        r = p["disp"]
        size = rng.negative_binomial(r, r / (r + p["mean"]), n_days)
    if kind in ("intermittent", "lumpy"):
        size = np.maximum(size, 1)
    return np.where(active, size, 0)


def generate(
    out_dir: str | pathlib.Path,
    stations_per_segment: int = 3,
    seed: int = 42,
) -> dict:
    """Write raw/trips/*.csv and raw/external/*.csv under out_dir; return a manifest."""
    rng = np.random.default_rng(seed)
    out = pathlib.Path(out_dir)
    trips_dir, ext_dir = out / config.LOCAL_RAW_TRIPS, out / config.LOCAL_RAW_EXTERNAL
    trips_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(config.START, config.END, freq="D")
    stations: list[tuple[str, str, str]] = []  # (id, name, kind)
    sid = 40000
    for kind in PROFILES:
        for i in range(stations_per_segment):
            stations.append((str(sid), f"Synthetic {kind.title()} St & {i + 1}th St NW", kind))
            sid += 1
    new_station = (str(sid), "Brand New Station", "smooth")  # opens in March 2026: not eligible

    rows = []
    ride = 0

    def add_trips(st_id: str, st_name: str, day: pd.Timestamp, n: int) -> None:
        nonlocal ride
        secs = rng.integers(5 * 3600, 23 * 3600, n)
        dur = rng.integers(120, 3600, n)
        for s, d in zip(secs, dur):
            start = day + pd.Timedelta(seconds=int(s), milliseconds=int(rng.integers(0, 1000)))
            end = start + pd.Timedelta(seconds=int(d))
            rows.append(_row(f"{ride:016X}", st_id, st_name, start, end, rng))
            ride += 1

    truth = {}
    for st_id, st_name, kind in stations:
        counts = _daily_counts(kind, len(days), rng)
        truth[st_id] = kind
        for day, n in zip(days, counts):
            if n:
                add_trips(st_id, st_name, day, int(n))
    for day in days[days >= pd.Timestamp("2026-03-01")]:
        add_trips(new_station[0], new_station[1], day, int(rng.poisson(10)))

    trips = pd.DataFrame(rows, columns=TRIP_COLUMNS)
    clean_rows = len(trips)

    # Dirty rows that silver must remove.
    dupes = trips.sample(25, random_state=seed)
    dockless = trips.sample(40, random_state=seed + 1).copy()
    dockless["ride_id"] = [f"D{i:015X}" for i in range(len(dockless))]
    dockless[["start_station_id", "start_station_name"]] = ""
    short = trips.sample(30, random_state=seed + 2).copy()
    short["ride_id"] = [f"S{i:015X}" for i in range(len(short))]
    short["ended_at"] = short["started_at"]
    trips = pd.concat([trips, dupes, dockless, short], ignore_index=True)

    trips["month"] = trips["started_at"].str[:7].str.replace("-", "")
    for ym, part in trips.groupby("month"):
        part.drop(columns="month").sample(frac=1, random_state=seed).to_csv(
            trips_dir / f"{ym}.csv", index=False)

    # External files in the same layout as the real scripts produce.
    gbfs = pd.DataFrame(
        {
            "station_id": [f"gbfs-{s[0]}" for s in stations] + [f"gbfs-{new_station[0]}"],
            "short_name": [s[0] for s in stations] + [new_station[0]],
            "name": [s[1] for s in stations] + [new_station[1]],
            "lat": rng.uniform(38.8, 39.0, len(stations) + 1),
            "lon": rng.uniform(-77.1, -76.9, len(stations) + 1),
            "capacity": rng.integers(8, 30, len(stations) + 1),
        }
    )
    gbfs = gbfs.iloc[:-2]  # two stations without a GBFS match, like the real data
    gbfs.to_csv(ext_dir / "stations.csv", index=False)

    doy = days.dayofyear.to_numpy()
    weather = pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "temperature_2m_mean": np.round(14 - 11 * np.cos(2 * np.pi * (doy - 15) / 365)
                                            + rng.normal(0, 3, len(days)), 1),
            "precipitation_sum": np.round(np.where(rng.random(len(days)) < 0.3,
                                                   rng.gamma(1.5, 4, len(days)), 0.0), 2),
        }
    )
    weather.to_csv(ext_dir / "weather_daily.csv", index=False)
    us = hol.country_holidays("US", subdiv="DC", years=[2025, 2026])
    pd.DataFrame({"date": [d.isoformat() for d in us], "holiday_name": list(us.values())}
                 ).sort_values("date").to_csv(ext_dir / "holidays.csv", index=False)

    return {
        "stations": truth,
        "new_station": new_station[0],
        "clean_rows": clean_rows,
        "duplicates": len(dupes),
        "dockless": len(dockless),
        "short": len(short),
    }


def _row(ride_id: str, st_id: str, st_name: str, start: pd.Timestamp, end: pd.Timestamp,
         rng: np.random.Generator) -> list:
    """One trip row in the current Capital Bikeshare layout."""
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    return [
        ride_id,
        "electric_bike" if rng.random() < 0.6 else "classic_bike",
        start.strftime(fmt)[:-3],
        end.strftime(fmt)[:-3],
        st_name, st_id,
        "Somewhere Else", "49999",
        38.9, -77.03, 38.91, -77.02,
        "member" if rng.random() < 0.7 else "casual",
    ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate synthetic raw data.")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--stations-per-segment", type=int, default=3)
    a = ap.parse_args()
    print(generate(a.out, a.stations_per_segment))
