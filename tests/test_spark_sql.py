"""Run the Databricks SQL (bikecast.sql) on a local Spark session and compare with pandas.

Skipped automatically when PySpark or a Java runtime is not available. CI installs both.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from bikecast import config, sql
from bikecast.segmentation import segment_stations

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java runtime not available", allow_module_level=True)

pytestmark = pytest.mark.spark


def tn(name: str) -> str:
    """Local tests use temp views named after the logical table."""
    return name


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    s = (SparkSession.builder.master("local[2]").appName("bikecast-tests")
         .config("spark.sql.session.timeZone", "UTC")  # Databricks default
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.enabled", "false")
         .getOrCreate())
    yield s
    s.stop()


@pytest.fixture(scope="module")
def spark_layers(spark, synthetic_dir):
    """Build every layer with the notebook SQL, as temp views."""
    from pyspark.sql import functions as F

    raw = synthetic_dir / "raw"
    (spark.read.option("header", True).option("inferSchema", False)
     .csv(str(raw / "trips" / "*.csv"))
     .withColumn("source_file", F.col("_metadata.file_path"))  # same as notebook 01
     .createOrReplaceTempView("bronze_trips"))
    for view, file in [("gbfs_stations_raw", "stations.csv"), ("weather_raw", "weather_daily.csv"),
                       ("holidays_raw", "holidays.csv")]:
        (spark.read.option("header", True).option("inferSchema", False)
         .csv(str(raw / "external" / file)).createOrReplaceTempView(view))

    out = {}
    for name, fn in [("silver_trips", sql.silver_trips), ("dim_stations", sql.dim_stations),
                     ("dim_calendar", sql.dim_calendar), ("station_daily", sql.station_daily),
                     ("station_segments", sql.station_segments)]:
        df = spark.sql(fn(tn))
        df.createOrReplaceTempView(name)
        out[name] = df
    out["removed"] = spark.sql(sql.silver_removal_counts(tn)).first().asDict()
    out["gold_checks"] = spark.sql(sql.gold_checks(tn)).first().asDict()
    return out


def test_silver_matches_pandas(spark_layers, layers):
    removed = {k: int(v) for k, v in spark_layers["removed"].items()}
    assert removed == layers["removed"]
    s = spark_layers["silver_trips"].select("ride_id", "duration_min").toPandas()
    p = layers["silver"][["ride_id", "duration_min"]].astype({"ride_id": str})
    m = s.merge(p, on="ride_id", suffixes=("_spark", "_pd"))
    assert len(m) == len(s) == len(p)
    assert (m.duration_min_spark - m.duration_min_pd).abs().max() < 1e-9


def test_dim_stations_matches_pandas(spark_layers, layers):
    s = spark_layers["dim_stations"].toPandas().sort_values("station_id").reset_index(drop=True)
    p = layers["dim"].astype({"station_id": str, "station_name": str})
    assert list(s.station_id) == list(p.station_id)
    assert list(s.station_name) == list(p.station_name)
    assert list(s.total_trips) == list(p.total_trips)
    assert s.lat.notna().sum() == p.lat.notna().sum()


def test_calendar_matches_pandas(spark_layers, layers):
    s = spark_layers["dim_calendar"].toPandas().sort_values("date").reset_index(drop=True)
    p = layers["cal"]
    assert len(s) == config.N_DAYS
    assert list(s.dow) == list(p.dow)
    assert list(s.is_weekend) == list(p.is_weekend)
    assert list(s.is_holiday) == list(p.is_holiday)
    assert (s.temp_mean.to_numpy() == p.temp_mean.to_numpy()).all()


def test_station_daily_matches_pandas(spark_layers, layers):
    s = (spark_layers["station_daily"].select("station_id", "date", "trips").toPandas()
         .assign(date=lambda d: pd.to_datetime(d.date))
         .sort_values(["station_id", "date"]).reset_index(drop=True))
    p = layers["daily"][["station_id", "date", "trips"]].astype({"station_id": str})
    pd.testing.assert_frame_equal(s, p.reset_index(drop=True), check_dtype=False)
    g = spark_layers["gold_checks"]
    assert g["days_per_station"] == config.N_DAYS and g["total_trips"] == p.trips.sum()


def test_segments_sql_equal_python(spark_layers, layers):
    s = spark_layers["station_segments"].toPandas().sort_values("station_id").reset_index(drop=True)
    p = segment_stations(layers["daily"]).astype({"station_id": str})
    assert list(s.station_id) == list(p.station_id)
    assert list(s.segment) == list(p.segment)
    assert (s.adi - p.adi).abs().max() < 1e-9 and (s.cv2 - p.cv2).abs().max() < 1e-9
