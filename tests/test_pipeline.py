"""Bronze -> silver -> gold on synthetic data: row counts, zero-filling and the gold checks."""

import pandas as pd
import pytest

from bikecast import config, pipeline


def test_bronze_has_one_file_per_month_and_lineage(layers):
    counts = pipeline.bronze_file_counts(layers["bronze"])
    assert len(counts) == 18
    assert {"source_file", "ingested_at"} <= set(layers["bronze"].columns)


def test_silver_removes_exactly_the_dirty_rows(layers, manifest):
    r = layers["removed"]
    assert r["duplicates"] == manifest["duplicates"]
    assert r["no_station"] == manifest["dockless"]
    assert r["bad_duration"] == manifest["short"]
    assert r["silver_rows"] == manifest["clean_rows"]
    assert r["bronze_rows"] == r["silver_rows"] + r["duplicates"] + r["no_station"] + \
        r["out_of_period"] + r["bad_duration"]


def test_silver_types_and_rules(layers):
    s = layers["silver"]
    assert s.ride_id.is_unique
    assert pd.api.types.is_datetime64_any_dtype(s.started_at)
    assert s.duration_min.between(1, 1440).all()
    assert (s.start_date == s.started_at.dt.normalize()).all()


def test_clean_trips_rules_on_handmade_rows():
    bronze = pd.DataFrame({
        "ride_id": ["a", "a", "b", "c", "d", "e"],
        "rideable_type": "classic_bike", "member_casual": "member",
        "start_station_id": [" 1 ", "1", None, "2", "3", "4"],
        "start_station_name": "X",
        "started_at": ["2025-03-01 10:00:00.5", "2025-03-01 11:00:00", "2025-03-01 10:00:00",
                       "2025-03-01 10:00:00", "2024-12-31 23:00:00", "not a date"],
        "ended_at": ["2025-03-01 10:10:00", "2025-03-01 11:10:00", "2025-03-01 10:10:00",
                     "2025-03-01 10:00:30", "2024-12-31 23:30:00", "2025-03-01 10:00:00"],
    })
    silver, removed = pipeline.clean_trips(bronze)
    assert list(silver.ride_id) == ["a"]
    assert silver.start_station_id.iloc[0] == "1"  # trimmed
    assert silver.started_at.iloc[0] == pd.Timestamp("2025-03-01 10:00:00.5")  # earliest kept
    assert removed == {"bronze_rows": 6, "duplicates": 1, "no_station": 1, "out_of_period": 2,
                       "bad_duration": 1, "silver_rows": 1}


def test_dim_stations_lifetime_and_gbfs_match(layers, manifest):
    dim = layers["dim"]
    assert len(dim) == len(manifest["stations"]) + 1
    assert dim.station_id.is_unique
    assert dim.lat.notna().sum() == len(dim) - 2  # two stations without GBFS match
    new = dim.set_index("station_id").loc[manifest["new_station"]]
    assert new.first_trip_date >= pd.Timestamp("2026-03-01")


def test_dim_stations_takes_most_frequent_name():
    silver = pd.DataFrame({
        "ride_id": list("abc"), "start_station_id": ["1", "1", "1"],
        "start_station_name": ["Old name", "New name", "New name"],
        "start_date": pd.to_datetime(["2025-01-01", "2025-06-01", "2026-06-01"]),
    })
    dim = pipeline.build_dim_stations(silver)
    assert dim.station_name.iloc[0] == "New name" and dim.total_trips.iloc[0] == 3


def test_calendar_has_546_days_spark_dow_and_flags(layers):
    cal = layers["cal"]
    assert len(cal) == config.N_DAYS
    assert cal[["temp_mean", "precip_mm"]].notna().all().all()
    jan1 = cal.set_index("date").loc["2025-01-01"]  # Wednesday, holiday
    assert jan1.dow == 4 and not jan1.is_weekend and jan1.is_holiday
    assert cal.set_index("date").loc["2025-01-05"].dow == 1  # Sunday


def test_gold_zero_filled_and_complete(layers, manifest):
    daily = layers["daily"]
    checks = pipeline.gold_checks(daily)
    assert checks["stations"] == len(manifest["stations"])  # new station excluded
    assert checks["days_per_station"] == config.N_DAYS
    assert manifest["new_station"] not in set(daily.station_id)
    assert (daily.trips == 0).any() and checks["pct_zero_days"] > 0
    # No trips lost in the join for eligible stations.
    silver = layers["silver"]
    assert checks["total_trips"] == silver.start_station_id.isin(set(daily.station_id)).sum()
    assert not daily.duplicated(["station_id", "date"]).any()


def test_model_frame_format(layers):
    df = pipeline.model_frame(layers["daily"], layers["segs"])
    assert list(df.columns[:3]) == ["unique_id", "ds", "y"]
    assert df.y.dtype == float and df.is_weekend.isin([0, 1]).all()


def test_spark_dayofweek():
    s = pd.Series(pd.to_datetime(["2026-06-07", "2026-06-08", "2026-06-13"]))  # Sun, Mon, Sat
    assert list(pipeline.spark_dayofweek(s)) == [1, 2, 7]


@pytest.mark.slow
def test_run_local_end_to_end(synthetic_dir, tmp_path):
    import shutil

    from bikecast.run_local import run

    work = tmp_path / "data"
    shutil.copytree(synthetic_dir / "raw", work / "raw")
    info = run(work, figures_dir=tmp_path / "fig", fast=True, n_jobs=1)
    out = work / "outputs"
    for name in ("station_daily", "station_segments", "forecasts", "metrics", "metrics_summary",
                 "future_forecast"):
        assert (out / f"{name}.parquet").exists()
    assert (out / "results.md").read_text(encoding="utf-8").count("|") > 20
    assert len(list((tmp_path / "fig").glob("*.png"))) == 6
    assert info["cutoffs"] == ["2026-06-02", "2026-06-09", "2026-06-16", "2026-06-23"]
    fc = pd.read_parquet(out / "forecasts.parquet")
    assert set(fc.model) == set(config.MODELS)
    assert fc.groupby(["station_id", "model"]).size().eq(28).all()  # same 28 test days each
    fut = pd.read_parquet(out / "future_forecast.parquet")
    assert fut.date.min() == pd.Timestamp("2026-07-01") and (fut.p10 <= fut.p90).all()
