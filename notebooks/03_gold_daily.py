# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: daily demand per station
# MAGIC **The most important detail:** days with zero trips must exist as rows with `trips = 0`.
# MAGIC Only stations that existed for the whole period are kept (a new station would look
# MAGIC intermittent just because it is new).

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


def tn(name: str) -> str:
    return config.t(name, CATALOG, SCHEMA)


# COMMAND ----------

spark.sql(f"CREATE OR REPLACE TABLE {tn('station_daily')} AS {sql.station_daily(tn)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC 1. exactly 546 days per station; 2. total trips equal silver for eligible stations;
# MAGIC 3. share of zero days. Quote these three numbers in the README.

# COMMAND ----------

checks = spark.sql(sql.gold_checks(tn)).first()
display(spark.sql(sql.gold_checks(tn)))

silver_eligible = spark.sql(f"""
  SELECT COUNT(*) AS n FROM {tn('silver_trips')}
  WHERE start_station_id IN (SELECT DISTINCT station_id FROM {tn('station_daily')})
""").first().n

assert checks.days_per_station == config.N_DAYS, checks
assert checks.total_trips == silver_eligible, (checks.total_trips, silver_eligible)
print(f"{checks.stations} stations x {config.N_DAYS} days, "
      f"{100 * checks.pct_zero_days:.1f}% of station-days have zero trips")

# COMMAND ----------

# Optional Delta maintenance (small table: not needed for speed, shows the command)
spark.sql(f"OPTIMIZE {tn('station_daily')} ZORDER BY (station_id)")
