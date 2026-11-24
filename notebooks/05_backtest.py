# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Backtest: five methods on the same four cutoffs
# MAGIC Seasonal Naive (baseline), Croston, TSB (statsforecast), zero-inflated Poisson (statsmodels)
# MAGIC and our Markov chain + Negative Binomial model. 4 windows of 7 days ending 30 June 2026.
# MAGIC TSB alphas and the Markov window are tuned on the **first two cutoffs only**.

# COMMAND ----------

# MAGIC %pip install statsforecast statsmodels
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bikeshare")
dbutils.widgets.dropdown("tune", "true", ["true", "false"])

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
import pandas as pd

from bikecast import backtest as B
from bikecast import config
from bikecast.markov import make_markov_fit_predict
from bikecast.zip_model import zip_fit_predict_flag

CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
TUNE = dbutils.widgets.get("tune") == "true"


def tn(name: str) -> str:
    return config.t(name, CATALOG, SCHEMA)


# COMMAND ----------

# Gold -> pandas (about 750 stations x 546 days: small enough for the driver)
df = spark.sql(f"""
  SELECT d.station_id AS unique_id, d.date AS ds, CAST(d.trips AS DOUBLE) AS y,
         CAST(d.is_weekend AS INT) AS is_weekend, CAST(d.is_holiday AS INT) AS is_holiday,
         d.temp_mean, d.precip_mm, s.segment
  FROM {tn('station_daily')} d
  JOIN {tn('station_segments')} s USING (station_id)
""").toPandas()
df["ds"] = pd.to_datetime(df["ds"])
df["unique_id"] = df["unique_id"].astype(str)
df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
cutoffs = B.rolling_cutoffs(config.END)
print(df.unique_id.nunique(), "stations;", "cutoffs:", [str(c.date()) for c in cutoffs])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning (first two cutoffs only)

# COMMAND ----------

if TUNE:
    alphas, tsb_scores = B.tune_tsb(df, cutoffs[:2])
    window, mk_scores = B.tune_markov_window(df, cutoffs[:2])
    display(tsb_scores.head(5))
    display(mk_scores)
else:
    alphas, window = config.DEFAULT_TSB_ALPHAS, config.DEFAULT_MARKOV_WINDOW
print("TSB alphas:", alphas, "| Markov window:", window)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library models, ZIP and Markov on the same cutoffs

# COMMAND ----------

cv = B.statsforecast_cv(df, tsb_alphas=alphas)
zip_fc = B.backtest(df, zip_fit_predict_flag, "ZIP", cutoffs, extra_name="fallback")
print("stations where ZIP used the fallback at least once:",
      int(zip_fc.groupby("unique_id")["ZIP_fallback"].max().sum()))
zip_fc = zip_fc.drop(columns="ZIP_fallback")
mk_fc = B.backtest(df, make_markov_fit_predict(window), "Markov", cutoffs)

# COMMAND ----------

fc = B.combine(cv, zip_fc, mk_fc)
res = B.score(fc, df)
summary = B.summarize(res)
table = B.mase_table(summary)
display(table.reset_index())
print(B.headline(summary))

# COMMAND ----------

# Save to Delta for the dashboard (overwrite = re-runnable)
for name, frame in {"forecasts": B.to_long(fc), "metrics": res, "metrics_summary": summary}.items():
    (spark.createDataFrame(frame).write.mode("overwrite").option("overwriteSchema", True)
     .saveAsTable(tn(name)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production output: next-week forecast after 30 June 2026 (PySpark, all stations in parallel)
# MAGIC The function is defined inside the cell because serverless workers may not see `src/`.

# COMMAND ----------

from pyspark.sql import functions as F

WINDOW, H = int(window), config.HORIZON


def per_station(pdf):
    import numpy as np
    import pandas as pd

    def transition_matrix(y):
        s = (y > 0).astype(int)
        prev, nxt = s[:-1], s[1:]
        n00 = np.sum((prev == 0) & (nxt == 0))
        n01 = np.sum((prev == 0) & (nxt == 1))
        n10 = np.sum((prev == 1) & (nxt == 0))
        n11 = np.sum((prev == 1) & (nxt == 1))
        p01 = (n01 + 1) / (n00 + n01 + 2)
        p11 = (n11 + 1) / (n10 + n11 + 2)
        return np.array([[1 - p01, p01], [1 - p11, p11]])

    def mean_size(y):
        z = y[y > 0] - 1
        return 1 + (z.mean() if len(z) else 0.0)

    pdf = pdf.sort_values("ds")
    y = pdf["y"].to_numpy(dtype=float)[-WINDOW:]
    P = transition_matrix(y)
    state = np.array([0.0, 1.0]) if y[-1] > 0 else np.array([1.0, 0.0])
    p = []
    for _ in range(H):
        state = state @ P
        p.append(state[1])
    p = np.array(p)
    days = pd.date_range(pd.Timestamp(pdf["ds"].max()) + pd.Timedelta(days=1), periods=H)
    return pd.DataFrame({"station_id": pdf["unique_id"].iloc[0], "date": days.date,
                         "y_hat": p * mean_size(y), "p_sale": p})


future_fc = (
    spark.table(tn("station_daily"))
    .join(spark.table(tn("station_segments")).select("station_id"), "station_id")
    .select(F.col("station_id").alias("unique_id"), F.col("date").alias("ds"),
            F.col("trips").cast("double").alias("y"))
    .groupBy("unique_id")
    .applyInPandas(per_station, schema="station_id string, date date, y_hat double, p_sale double")
)
future_fc.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(tn("future_forecast"))
display(spark.table(tn("future_forecast")).orderBy(F.desc("y_hat")).limit(20))
