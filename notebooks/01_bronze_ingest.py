# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingest
# MAGIC Reads the 18 monthly trip CSVs from the Unity Catalog Volume **as strings** (a bad row never
# MAGIC breaks the load) and writes them to `bronze_trips` with lineage columns.
# MAGIC Re-runnable: always `overwrite`.

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
from bikecast import config

CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{config.VOLUME}"
BRONZE = config.t("bronze_trips", CATALOG, SCHEMA)
print(BRONZE, VOLUME_PATH)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{config.VOLUME}")

# COMMAND ----------

from pyspark.sql import functions as F

raw = (
    spark.read.option("header", True)
    .option("inferSchema", False)
    .csv(f"{VOLUME_PATH}/trips/*.csv")
    .withColumn("source_file", F.col("_metadata.file_path"))
    .withColumn("ingested_at", F.current_timestamp())
)
(raw.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(BRONZE))

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** 18 rows, each with a few hundred thousand trips. A month with far fewer trips
# MAGIC probably failed to unzip properly.

# COMMAND ----------

counts = spark.sql(f"""
  SELECT source_file, COUNT(*) AS n
  FROM {BRONZE}
  GROUP BY source_file ORDER BY source_file
""")
display(counts)
n_files = counts.count()
assert n_files == 18, f"expected 18 monthly files, found {n_files}"
