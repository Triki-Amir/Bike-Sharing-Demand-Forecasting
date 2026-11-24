# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Segmentation with ADI and CV²
# MAGIC Computed on the **training period only** (up to 2 June 2026) so the test days do not leak.
# MAGIC Thresholds 1.32 / 0.49 from Syntetos & Boylan (2005).

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
from bikecast import config, plots, sql
from bikecast.segmentation import segment_stations

CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")


def tn(name: str) -> str:
    return config.t(name, CATALOG, SCHEMA)


# COMMAND ----------

spark.sql(f"CREATE OR REPLACE TABLE {tn('station_segments')} AS {sql.station_segments(tn)}")
display(spark.sql(f"""
  SELECT segment, COUNT(*) AS stations, ROUND(AVG(mean_daily_trips), 2) AS mean_daily_trips,
         ROUND(AVG(pct_zero_days), 3) AS pct_zero_days
  FROM {tn('station_segments')} GROUP BY segment ORDER BY segment
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL and Python give the same segments

# COMMAND ----------

daily = spark.table(tn("station_daily")).select("station_id", "date", "trips").toPandas()
segs_sql = spark.table(tn("station_segments")).toPandas()
segs_py = segment_stations(daily)
merged = segs_sql.merge(segs_py, on="station_id", suffixes=("_sql", "_py"))
mismatch = merged[merged.segment_sql != merged.segment_py]
print(f"{len(merged)} stations compared, {len(mismatch)} mismatches")
assert len(merged) == len(segs_sql) == len(segs_py) and mismatch.empty

# COMMAND ----------

# MAGIC %md
# MAGIC ## Charts

# COMMAND ----------

display(plots.adi_cv2_scatter(segs_sql))

# COMMAND ----------

display(plots.segment_bars(segs_sql, daily))

# COMMAND ----------

display(plots.example_series(daily, segs_sql))

# COMMAND ----------

# Optional map: stations coloured by segment (quiet stations sit at the edges of the network)
stations = spark.table(tn("dim_stations")).select("station_id", "station_name", "lat", "lon").toPandas()
geo = segs_sql.merge(stations, on="station_id").dropna(subset=["lat", "lon"])
try:
    import plotly.express as px

    fig = px.scatter_mapbox(geo, lat="lat", lon="lon", color="segment", hover_name="station_name",
                            color_discrete_map=plots.SEGMENT_COLORS, zoom=10, height=550,
                            mapbox_style="carto-positron")
    fig.show()
except ImportError:
    print("plotly not available; skipping the map")
