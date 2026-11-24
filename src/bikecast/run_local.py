"""Run the whole project locally: bronze -> silver -> gold -> segmentation -> backtest -> report.

Usage:
    python -m bikecast.run_local                       # real data in ./data (after the scripts)
    python -m bikecast.run_local --synthetic --fast    # synthetic data, used by CI

Outputs go to ``<data-dir>/outputs`` (tables as parquet, results.json, results.md) and the
figures to ``--figures-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd

from bikecast import backtest as B
from bikecast import config, pipeline, plots, synthetic
from bikecast.markov import make_markov_fit_predict, markov_forecast, prediction_interval
from bikecast.segmentation import segment_stations
from bikecast.zip_model import zip_fit_predict_flag


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def future_forecast(df: pd.DataFrame, window: int, h: int = config.HORIZON) -> pd.DataFrame:
    """Production output: Markov forecast for the h days after the last date, with 80% PI."""
    rows = []
    for sid, g in df.groupby("unique_id"):
        y = g.sort_values("ds")["y"].to_numpy()
        y_hat, p = markov_forecast(y, h=h, window=window)
        lo, hi = prediction_interval(y, h=h, window=window, n_paths=1000, seed=0)
        days = pd.date_range(pd.Timestamp(g.ds.max()) + pd.Timedelta(days=1), periods=h)
        rows.append(pd.DataFrame({"station_id": sid, "date": days, "y_hat": y_hat, "p_sale": p,
                                  "p10": lo, "p90": hi}))
    return pd.concat(rows, ignore_index=True)


def results_markdown(table: pd.DataFrame, summary: pd.DataFrame, info: dict) -> str:
    """Markdown block with the MASE table and the key numbers."""
    def fmt(v):
        return "" if pd.isna(v) else f"{v:.2f}"

    models = [m for m in config.MODELS if m in table.columns]
    best = table[models].idxmin(axis=1)
    head = "| Segment | Stations | " + " | ".join(plots.MODEL_LABELS[m] for m in models) + " |"
    sep = "|---|---:|" + "---:|" * len(models)
    lines = [head, sep]
    for seg, row in table.iterrows():
        cells = [f"**{fmt(row[m])}**" if best[seg] == m else fmt(row[m]) for m in models]
        lines.append(f"| {seg} | {int(row['stations'])} | " + " | ".join(cells) + " |")
    g = info["gold"]
    bias = summary.pivot(index="segment", columns="model", values="bias_mean").reindex(
        table.index)[models]
    bias_lines = [head.replace("Stations | ", ""), sep.replace("---:|", "", 1)]
    for seg, row in bias.iterrows():
        bias_lines.append(f"| {seg} | " + " | ".join(f"{100 * row[m]:+.1f}%" for m in models)
                          + " |")
    return "\n".join([
        f"Data: {g['stations']} stations x {g['days_per_station']:.0f} days, "
        f"{100 * g['pct_zero_days']:.1f}% of station-days have zero trips "
        f"({g['total_trips']:,} trips).",
        "",
        "Median MASE per segment (lower is better, below 1 beats the in-sample seasonal naive; "
        "best model in bold):",
        "",
        *lines,
        "",
        "Mean bias per segment (positive = over-forecast):",
        "",
        *bias_lines,
    ])


def run(
    data_dir: str | pathlib.Path,
    figures_dir: str | pathlib.Path | None = None,
    fast: bool = False,
    n_jobs: int = 1,
    max_mean_daily: float | None = None,
) -> dict:
    """Run every layer and the backtest; return a dict with the key results."""
    data_dir = pathlib.Path(data_dir)
    out = data_dir / config.LOCAL_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    ext = data_dir / config.LOCAL_RAW_EXTERNAL
    info: dict = {"fast": fast}
    t0 = time.time()

    # ---------------------------------------------------------------- bronze / silver
    log("bronze: reading trip CSVs")
    bronze = pipeline.read_bronze(data_dir / config.LOCAL_RAW_TRIPS)
    info["bronze_files"] = pipeline.bronze_file_counts(bronze).set_index("source_file")["n"].to_dict()
    log(f"bronze: {len(bronze):,} rows from {len(info['bronze_files'])} files")
    silver, removed = pipeline.clean_trips(bronze)
    del bronze
    info["silver_removed"] = removed
    log(f"silver: {removed}")

    gbfs = pd.read_csv(ext / "stations.csv", dtype={"short_name": str, "station_id": str})
    dim_st = pipeline.build_dim_stations(silver, gbfs)
    cal = pipeline.build_calendar(pd.read_csv(ext / "weather_daily.csv"),
                                  pd.read_csv(ext / "holidays.csv"))
    info["stations_total"] = len(dim_st)
    info["stations_gbfs_match"] = int(dim_st["lat"].notna().sum())
    info["calendar_rows"] = len(cal)
    info["calendar_missing_weather"] = int(cal[["temp_mean", "precip_mm"]].isna().any(axis=1).sum())
    log(f"silver: {len(dim_st)} stations ({info['stations_gbfs_match']} GBFS matches), "
        f"calendar {len(cal)} rows, {info['calendar_missing_weather']} missing weather")

    # ---------------------------------------------------------------- gold
    daily = pipeline.build_station_daily(silver, dim_st, cal)
    info["gold"] = pipeline.gold_checks(daily)
    eligible = set(daily.station_id)
    info["silver_trips_eligible"] = int(silver.start_station_id.isin(eligible).sum())
    del silver
    log(f"gold: {info['gold']}")

    segs = segment_stations(daily)
    info["segments_all"] = segs.segment.value_counts().to_dict()
    if max_mean_daily is not None:
        segs = segs[segs.mean_daily_trips < max_mean_daily]
        info["quiet_filter"] = f"mean_daily_trips < {max_mean_daily}"
    info["segments"] = segs.segment.value_counts().to_dict()
    log(f"segments: {info['segments']}")

    # ---------------------------------------------------------------- backtest
    df = pipeline.model_frame(daily, segs)
    cutoffs = B.rolling_cutoffs(config.END)
    info["cutoffs"] = [str(c.date()) for c in cutoffs]
    tuning_cutoffs = cutoffs[:2]

    if fast:
        alphas, window = config.DEFAULT_TSB_ALPHAS, config.DEFAULT_MARKOV_WINDOW
    else:
        log("tuning TSB alphas on the first two cutoffs")
        alphas, tsb_scores = B.tune_tsb(df, tuning_cutoffs, n_jobs=n_jobs)
        log("tuning Markov window on the first two cutoffs")
        window, mk_scores = B.tune_markov_window(df, tuning_cutoffs)
        info["tuning"] = {"tsb": tsb_scores.head(5).to_dict("records"),
                          "markov": mk_scores.to_dict("records")}
    info["tsb_alphas"], info["markov_window"] = list(alphas), int(window)
    log(f"chosen: TSB alphas={alphas}, Markov window={window}")

    log("statsforecast: Seasonal Naive, Croston, TSB")
    cv = B.statsforecast_cv(df, tsb_alphas=alphas, n_jobs=n_jobs)
    log("ZIP backtest")
    zip_fc = B.backtest(df, zip_fit_predict_flag, "ZIP", cutoffs, n_jobs=n_jobs,
                        extra_name="fallback")
    fb = zip_fc.groupby("unique_id")["ZIP_fallback"].max()
    info["zip_fallback_stations"] = int(fb.sum())
    info["zip_fallback_fits"] = int(zip_fc.groupby(["unique_id", "cutoff"])["ZIP_fallback"]
                                    .max().sum())
    zip_fc = zip_fc.drop(columns="ZIP_fallback")
    log("Markov backtest")
    mk_fc = B.backtest(df, make_markov_fit_predict(window), "Markov", cutoffs)

    fc = B.combine(cv, zip_fc, mk_fc)
    res = B.score(fc, df)
    summary = B.summarize(res)
    table = B.mase_table(summary)
    info["mase_table"] = table.round(4).reset_index().to_dict("records")
    info["headline"] = B.headline(summary)
    fc_long = B.to_long(fc)
    fut = future_forecast(df, window)
    log("results:\n" + table.round(3).to_string())

    # ---------------------------------------------------------------- save
    for name, frame in {"dim_stations": dim_st, "dim_calendar": cal, "station_daily": daily,
                        "station_segments": segs, "forecasts": fc_long, "metrics": res,
                        "metrics_summary": summary, "future_forecast": fut}.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    md = results_markdown(table, summary, info)
    (out / "results.md").write_text(md, encoding="utf-8")
    if figures_dir is not None:
        plots.all_figures(figures_dir, daily, segs, summary, fc_long)
        log(f"figures written to {figures_dir}")
    info["runtime_s"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(info, indent=2, default=_json_default),
                                      encoding="utf-8")
    log(f"done in {info['runtime_s']} s; outputs in {out}")
    return info


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=config.LOCAL_DATA_DIR)
    ap.add_argument("--figures-dir", default=None, help="where to write PNG charts")
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic raw data first")
    ap.add_argument("--stations-per-segment", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="skip tuning (default parameters)")
    ap.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--max-mean-daily", type=float, default=None,
                    help="keep only stations with mean daily trips below this value")
    a = ap.parse_args()
    if a.synthetic:
        log(f"generating synthetic data in {a.data_dir}")
        synthetic.generate(a.data_dir, a.stations_per_segment)
    info = run(a.data_dir, a.figures_dir, a.fast, a.n_jobs, a.max_mean_daily)
    print(json.dumps(info["headline"], indent=2))


if __name__ == "__main__":
    main()
