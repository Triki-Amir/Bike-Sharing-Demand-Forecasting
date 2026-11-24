# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Report: results table and charts
# MAGIC Reads `metrics`, `metrics_summary`, `forecasts` and `station_segments` from Delta.

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
from bikecast import backtest as B
from bikecast import config, plots

CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")


def tn(name: str) -> str:
    return config.t(name, CATALOG, SCHEMA)


summary = spark.table(tn("metrics_summary")).toPandas()
segs = spark.table(tn("station_segments")).toPandas()
fc_long = spark.table(tn("forecasts")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Median MASE: rows = segments, columns = models (lower is better, < 1 beats naive)

# COMMAND ----------

display(B.mase_table(summary).round(3).reset_index())
print(B.headline(summary))

# COMMAND ----------

display(plots.mase_grouped_bars(summary))

# COMMAND ----------

display(plots.bias_bars(summary))

# COMMAND ----------

quiet = segs[segs.segment.isin(["lumpy", "intermittent"])].sort_values("mean_daily_trips")
station = quiet.iloc[len(quiet) // 2].station_id if len(quiet) else segs.station_id.iloc[0]
display(plots.station_closeup(fc_long, station))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard datasets (Dashboards → Create dashboard → Data tab)
# MAGIC ```sql
# MAGIC SELECT segment, model, percentile(mase, 0.5) AS mase_median FROM workspace.bikeshare.metrics GROUP BY ALL
# MAGIC SELECT segment, COUNT(*) AS stations FROM workspace.bikeshare.station_segments GROUP BY segment
# MAGIC SELECT * FROM workspace.bikeshare.forecasts WHERE station_id = :station
# MAGIC ```
