# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: clean trips, stations and calendar
# MAGIC Casts types, removes duplicates and invalid rows (no station, outside the period, duration
# MAGIC outside 1 min – 24 h), then builds `dim_stations` and `dim_calendar`.
# MAGIC The SQL lives in `src/bikecast/sql.py` and is tested in CI on a local Spark session.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bikeshare")

# COMMAND ----------

import os
import sys


def _repo_root() -> str:
    """Repo root from the notebook location (works in Git folders and bundle deployments)."""
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        nb = nb if nb.startswith("/Workspace") else "/Workspace" + nb
        return nb.rsplit("/notebooks/", 1)[0]
    except Exception:
        return os.path.dirname(os.getcwd())


sys.path.insert(0, f"{_repo_root()}/src")
from bikecast import config, sql

CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
EXTERNAL = f"/Volumes/{CATALOG}/{SCHEMA}/{config.VOLUME}/external"


def tn(name: str) -> str:
    """Raw external files are temp views; everything else is a Unity Catalog table."""
    return name if name.endswith("_raw") else config.t(name, CATALOG, SCHEMA)


# COMMAND ----------

# External CSVs -> temporary views (read as strings, cast in SQL)
for view, file in [("gbfs_stations_raw", "stations.csv"), ("weather_raw", "weather_daily.csv"),
                   ("holidays_raw", "holidays.csv")]:
    (spark.read.option("header", True).option("inferSchema", False)
     .csv(f"{EXTERNAL}/{file}").createOrReplaceTempView(view))

# COMMAND ----------

spark.sql(f"CREATE OR REPLACE TABLE {tn('silver_trips')} AS {sql.silver_trips(tn)}")
spark.sql(f"CREATE OR REPLACE TABLE {tn('dim_stations')} AS {sql.dim_stations(tn)}")
spark.sql(f"CREATE OR REPLACE TABLE {tn('dim_calendar')} AS {sql.dim_calendar(tn)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality checks (screenshot these for the report)

# COMMAND ----------

removed = spark.sql(sql.silver_removal_counts(tn))
display(removed)  # rows removed between bronze and silver, and why
r = removed.first().asDict()
assert r["silver_rows"] == spark.table(tn("silver_trips")).count()

# COMMAND ----------

display(spark.sql(f"""
  SELECT COUNT(*) AS stations, COUNT(lat) AS with_gbfs_match,
         ROUND(100 * COUNT(lat) / COUNT(*), 1) AS pct_matched
  FROM {tn('dim_stations')}
"""))

# COMMAND ----------

cal = spark.sql(f"""
  SELECT COUNT(*) AS n_days,
         SUM(CASE WHEN temp_mean IS NULL OR precip_mm IS NULL THEN 1 ELSE 0 END) AS missing_weather,
         SUM(CASE WHEN is_holiday THEN 1 ELSE 0 END) AS holidays
  FROM {tn('dim_calendar')}
""").first()
print(cal)
assert cal.n_days == config.N_DAYS, cal
assert cal.missing_weather == 0, cal
