"""Local (pandas) implementation of the bronze -> silver -> gold layers.

It mirrors the Spark SQL in ``bikecast.sql`` / the Databricks notebooks row for row, so the
whole project can run and be tested on a laptop or in CI without Databricks.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from bikecast import config

TRIP_COLUMNS = [
    "ride_id", "rideable_type", "started_at", "ended_at", "start_station_name",
    "start_station_id", "end_station_name", "end_station_id", "start_lat", "start_lng",
    "end_lat", "end_lng", "member_casual",
]
SILVER_SOURCE_COLUMNS = [
    "ride_id", "rideable_type", "started_at", "ended_at", "start_station_name",
    "start_station_id", "member_casual",
]


# --------------------------------------------------------------------------- bronze
def read_bronze(trips_dir: str | pathlib.Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read every monthly CSV as strings, adding source_file and ingested_at."""
    files = sorted(pathlib.Path(trips_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {trips_dir}")
    cols = columns or SILVER_SOURCE_COLUMNS
    now = pd.Timestamp.now()
    parts = []
    for f in files:
        part = pd.read_csv(f, usecols=lambda c: c in cols, dtype="string[pyarrow]",
                           keep_default_na=False, na_values=[""])
        for c in cols:
            if c not in part.columns:
                part[c] = pd.NA
        part = part[cols]
        part["source_file"] = f.name
        part["ingested_at"] = now
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def bronze_file_counts(bronze: pd.DataFrame) -> pd.DataFrame:
    """Rows per source file (one per month expected)."""
    return bronze.groupby("source_file").size().rename("n").reset_index()


# --------------------------------------------------------------------------- silver
def clean_trips(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Type, deduplicate and filter trips; returns (silver_trips, removal counts)."""
    df = pd.DataFrame(
        {
            "ride_id": bronze["ride_id"],
            "rideable_type": bronze["rideable_type"],
            "member_casual": bronze["member_casual"],
            "start_station_id": bronze["start_station_id"].str.strip(),
            "start_station_name": bronze["start_station_name"],
            "started_at": pd.to_datetime(bronze["started_at"], format="ISO8601", errors="coerce"),
            "ended_at": pd.to_datetime(bronze["ended_at"], format="ISO8601", errors="coerce"),
        }
    )
    n_bronze = len(df)

    # ROW_NUMBER() OVER (PARTITION BY ride_id ORDER BY started_at): Spark sorts NULLs first.
    df = df.sort_values("started_at", kind="stable", na_position="first")
    df = df.drop_duplicates("ride_id", keep="first")
    n_dupes = n_bronze - len(df)

    sid = df["start_station_id"]
    no_station = (sid.isna() | (sid == "")).fillna(True).astype(bool)
    start = pd.Timestamp(config.START)
    end = pd.Timestamp(config.END) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    in_period = df["started_at"].between(start, end).fillna(False).astype(bool)
    secs = (df["ended_at"].dt.floor("s") - df["started_at"].dt.floor("s")).dt.total_seconds()
    good_duration = secs.between(config.MIN_DURATION_S, config.MAX_DURATION_S).fillna(False)
    good_duration = good_duration.astype(bool)

    keep = ~no_station & in_period & good_duration
    removed = {
        "bronze_rows": n_bronze,
        "duplicates": n_dupes,
        "no_station": int(no_station.sum()),
        "out_of_period": int((~no_station & ~in_period).sum()),
        "bad_duration": int((~no_station & in_period & ~good_duration).sum()),
    }
    silver = df[keep].copy()
    silver["start_date"] = silver["started_at"].dt.normalize()
    silver["duration_min"] = secs[keep] / 60.0
    silver = silver.sort_values(["start_station_id", "started_at"]).reset_index(drop=True)
    removed["silver_rows"] = len(silver)
    return silver, removed


def build_dim_stations(silver: pd.DataFrame, gbfs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per station: most frequent name, first/last trip date, GBFS coordinates."""
    names = (
        silver.groupby(["start_station_id", "start_station_name"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["start_station_id", "n", "start_station_name"],
                     ascending=[True, False, True], kind="stable")
        .drop_duplicates("start_station_id")
        .rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"})
    )
    life = (
        silver.groupby("start_station_id")
        .agg(first_trip_date=("start_date", "min"), last_trip_date=("start_date", "max"),
             total_trips=("ride_id", "size"))
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
    )
    out = names[["station_id", "station_name"]].merge(life, on="station_id")
    if gbfs is not None and len(gbfs):
        g = gbfs.copy()
        g["short_name"] = g["short_name"].astype(str).str.strip()
        g["station_id"] = g["station_id"].astype(str)
        g = g.sort_values("station_id", kind="stable").drop_duplicates("short_name")
        g = g[["short_name", "lat", "lon", "capacity"]]
        out = out.merge(g, left_on="station_id", right_on="short_name", how="left").drop(
            columns="short_name")
    else:
        out[["lat", "lon", "capacity"]] = np.nan
    return out.sort_values("station_id").reset_index(drop=True)


def spark_dayofweek(dates: pd.Series) -> pd.Series:
    """Spark's dayofweek: 1 = Sunday ... 7 = Saturday."""
    return (pd.to_datetime(dates).dt.dayofweek + 1) % 7 + 1


def build_calendar(weather: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """One row per day of the study period with weekday, holiday flag and weather."""
    days = pd.DataFrame({"date": pd.date_range(config.START, config.END, freq="D")})
    w = weather.rename(columns={"temperature_2m_mean": "temp_mean",
                                "precipitation_sum": "precip_mm"}).copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w[["date", "temp_mean", "precip_mm"]].astype({"temp_mean": float, "precip_mm": float})
    h = pd.to_datetime(holidays["date"]).drop_duplicates()
    cal = days.merge(w, on="date", how="left")
    cal["dow"] = spark_dayofweek(cal["date"])
    cal["is_weekend"] = cal["dow"].isin([1, 7])
    cal["is_holiday"] = cal["date"].isin(h)
    return cal[["date", "dow", "is_weekend", "is_holiday", "temp_mean", "precip_mm"]]


# --------------------------------------------------------------------------- gold
def build_station_daily(
    silver: pd.DataFrame, dim_stations: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Station x day grid for stations alive the whole period, zero days filled with 0."""
    eligible = dim_stations[
        (dim_stations["first_trip_date"] <= pd.Timestamp(config.ELIGIBLE_FIRST_TRIP_BY))
        & (dim_stations["last_trip_date"] >= pd.Timestamp(config.ELIGIBLE_LAST_TRIP_FROM))
    ]["station_id"]
    counts = (
        silver.groupby(["start_station_id", "start_date"]).size().rename("trips").reset_index()
        .rename(columns={"start_station_id": "station_id", "start_date": "date"})
    )
    grid = pd.MultiIndex.from_product(
        [sorted(eligible), calendar["date"]], names=["station_id", "date"]
    ).to_frame(index=False)
    out = grid.merge(counts, on=["station_id", "date"], how="left")
    out["trips"] = out["trips"].fillna(0).astype(int)
    out = out.merge(calendar, on="date", how="inner")
    cols = ["station_id", "date", "trips", "dow", "is_weekend", "is_holiday", "temp_mean",
            "precip_mm"]
    return out[cols].sort_values(["station_id", "date"]).reset_index(drop=True)


def gold_checks(station_daily: pd.DataFrame) -> dict:
    """The three gold checks: stations, days per station, share of zero days, total trips."""
    n_st = station_daily["station_id"].nunique()
    return {
        "stations": int(n_st),
        "days_per_station": float(len(station_daily) / n_st) if n_st else 0.0,
        "total_trips": int(station_daily["trips"].sum()),
        "pct_zero_days": float((station_daily["trips"] == 0).mean()) if n_st else float("nan"),
    }


def model_frame(station_daily: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Gold joined to segments, in statsforecast format (unique_id, ds, y + features)."""
    df = station_daily.merge(segments[["station_id", "segment"]], on="station_id", how="inner")
    df = df.rename(columns={"station_id": "unique_id", "date": "ds", "trips": "y"})
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    for c in ("is_weekend", "is_holiday"):
        df[c] = df[c].astype(int)
    cols = ["unique_id", "ds", "y", "is_weekend", "is_holiday", "temp_mean", "precip_mm",
            "segment"]
    return df[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)
