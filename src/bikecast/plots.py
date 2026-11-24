"""Charts for the report, README and slides (matplotlib, static PNG)."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bikecast import config  # noqa: E402

# Validated categorical palette (fixed order, never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEGMENT_COLORS = dict(zip(config.SEGMENTS, PALETTE[:4]))
MODEL_COLORS = dict(zip(config.MODELS, PALETTE[:5]))
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
                                         "#fcfcfb")
MODEL_LABELS = {"SeasonalNaive": "Seasonal Naive", "CrostonClassic": "Croston", "TSB": "TSB",
                "ZIP": "ZIP", "Markov": "Markov + NegBin"}


def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Recessive grid and axes, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _fig(w: float = 8, h: float = 4.5, **kw) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, ax


def _save(fig: plt.Figure, path: str | pathlib.Path | None) -> plt.Figure:
    if path is not None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    return fig


def adi_cv2_scatter(segments: pd.DataFrame, path=None) -> plt.Figure:
    """ADI vs CV², log-log, one dot per station, coloured by segment, Syntetos-Boylan cuts."""
    fig, ax = _fig(7.5, 5.5)
    for seg in config.SEGMENTS:
        s = segments[segments.segment == seg]
        if s.empty:
            continue
        ax.scatter(s.adi, s.cv2.clip(lower=1e-3), s=22, color=SEGMENT_COLORS[seg], alpha=0.85,
                   edgecolor=SURFACE, linewidth=0.6, label=f"{seg} ({len(s)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(config.ADI_CUT, color=INK2, ls="--", lw=1)
    ax.axhline(config.CV2_CUT, color=INK2, ls="--", lw=1)
    ax.text(config.ADI_CUT, ax.get_ylim()[1], f" ADI = {config.ADI_CUT}", color=INK2, fontsize=8,
            va="top")
    ax.text(ax.get_xlim()[1], config.CV2_CUT, f"CV² = {config.CV2_CUT} ", color=INK2, fontsize=8,
            ha="right", va="bottom")
    ax.set_xlabel("ADI: average days between active days (log)")
    ax.set_ylabel("CV² of non-zero daily trips (log)")
    ax.set_title("Station segmentation (training period)", loc="left", fontsize=12)
    _style(ax, "both")
    ax.legend(frameon=False, fontsize=9, title="Segment (stations)", title_fontsize=9)
    return _save(fig, path)


def segment_bars(segments: pd.DataFrame, station_daily: pd.DataFrame, path=None) -> plt.Figure:
    """Two panels: number of stations per segment, and share of total trips per segment."""
    trips = station_daily.groupby("station_id")["trips"].sum()
    seg = segments.set_index("station_id")["segment"]
    order = [s for s in config.SEGMENTS if s in set(seg)]
    n = seg.value_counts().reindex(order)
    share = 100 * trips.reindex(seg.index).groupby(seg).sum().reindex(order) / trips.reindex(
        seg.index).sum()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), facecolor=SURFACE)
    for ax, vals, title, fmt in ((axes[0], n, "Stations per segment", "{:.0f}"),
                                 (axes[1], share, "Share of all trips (%)", "{:.1f}%")):
        bars = ax.bar(order, vals.values, color=[SEGMENT_COLORS[s] for s in order], width=0.6,
                      edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center",
                    va="bottom", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def example_series(station_daily: pd.DataFrame, segments: pd.DataFrame, path=None) -> plt.Figure:
    """Small multiples: one representative station per segment, daily trips over the period."""
    order = [s for s in config.SEGMENTS if s in set(segments.segment)]
    fig, axes = plt.subplots(len(order), 1, figsize=(9, 1.9 * len(order)), sharex=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, seg in zip(axes[:, 0], order):
        s = segments[segments.segment == seg]
        sid = s.iloc[(s.mean_daily_trips - s.mean_daily_trips.median()).abs().argmin()].station_id
        g = station_daily[station_daily.station_id == sid]
        ax.plot(pd.to_datetime(g.date), g.trips, color=SEGMENT_COLORS[seg], lw=1)
        ax.set_title(f"{seg}: station {sid}", loc="left", fontsize=10)
        ax.set_ylabel("trips/day", fontsize=8)
        _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def mase_grouped_bars(summary: pd.DataFrame, path=None) -> plt.Figure:
    """Median MASE per model, grouped by segment (lower is better; 1 = naive benchmark)."""
    order = [s for s in config.SEGMENTS if s in set(summary.segment)]
    models = [m for m in config.MODELS if m in set(summary.model)]
    piv = summary.pivot(index="segment", columns="model", values="mase_median").reindex(order)
    fig, ax = _fig(9, 4.8)
    width = 0.8 / len(models)
    x = np.arange(len(order))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, piv[m].values, width * 0.92,
               color=MODEL_COLORS[m], label=MODEL_LABELS[m], edgecolor=SURFACE, linewidth=1)
    ax.axhline(1.0, color=INK2, lw=1, ls=":")
    ax.set_xticks(x, order)
    ax.set_ylabel("median MASE (lower is better)")
    ax.set_title("Forecast accuracy by segment and model", loc="left", fontsize=12)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=len(models), loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    return _save(fig, path)


def bias_bars(summary: pd.DataFrame, segments=("intermittent", "lumpy"), path=None) -> plt.Figure:
    """Mean bias per model on the quiet segments (positive = over-forecast)."""
    sub = summary[summary.segment.isin(segments)]
    order = [s for s in segments if s in set(sub.segment)]
    models = [m for m in config.MODELS if m in set(sub.model)]
    piv = sub.pivot(index="segment", columns="model", values="bias_mean").reindex(order)
    fig, ax = _fig(8, 4.2)
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(order))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, 100 * piv[m].values, width * 0.92,
               color=MODEL_COLORS[m], label=MODEL_LABELS[m], edgecolor=SURFACE, linewidth=1)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("mean bias (%) — positive = over-forecast")
    ax.set_title("Bias on quiet stations", loc="left", fontsize=12)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=len(models), loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    return _save(fig, path)


def station_closeup(forecasts_long: pd.DataFrame, station_id: str, path=None) -> plt.Figure:
    """Actual June trips (bars) vs Markov forecast (line); p_sale in its own panel below."""
    g = forecasts_long[(forecasts_long.station_id == station_id)
                       & (forecasts_long.model == "Markov")].sort_values("date")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [2, 1]})
    d = pd.to_datetime(g.date)
    ax1.bar(d, g.y, color=AXIS, width=0.8, label="actual trips")
    ax1.plot(d, g.y_hat, color=MODEL_COLORS["Markov"], lw=2, marker="o", ms=4,
             label="Markov forecast")
    ax1.set_ylabel("trips")
    ax1.set_title(f"Station {station_id}: June 2026 backtest", loc="left", fontsize=12)
    ax1.legend(frameon=False, fontsize=9)
    ax2.plot(d, g.p_sale, color=MODEL_COLORS["Markov"], lw=2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(≥1 trip)")
    for ax in (ax1, ax2):
        _style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, path)


def all_figures(out_dir, station_daily, segments, summary, forecasts_long) -> list[pathlib.Path]:
    """Render every report figure into out_dir and return the paths."""
    out = pathlib.Path(out_dir)
    paths = {
        "adi_cv2_scatter.png": lambda p: adi_cv2_scatter(segments, p),
        "segment_bars.png": lambda p: segment_bars(segments, station_daily, p),
        "example_series.png": lambda p: example_series(station_daily, segments, p),
        "mase_by_segment.png": lambda p: mase_grouped_bars(summary, p),
        "bias_quiet_stations.png": lambda p: bias_bars(summary, path=p),
    }
    done = []
    for name, fn in paths.items():
        fn(out / name)
        done.append(out / name)
    quiet = segments[segments.segment.isin(["lumpy", "intermittent"])]
    pick = quiet if len(quiet) else segments
    sid = pick.sort_values("mean_daily_trips").iloc[len(pick) // 2].station_id
    station_closeup(forecasts_long, sid, out / "station_closeup.png")
    done.append(out / "station_closeup.png")
    return done
