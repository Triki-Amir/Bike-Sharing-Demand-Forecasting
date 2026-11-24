"""Build docs/architecture.png and docs/slides.pdf (5 slides) from the local run outputs.

Usage (after ``python -m bikecast.run_local --figures-dir docs/figures``):
    python scripts/make_docs.py [--outputs data/outputs] [--docs docs]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from bikecast import backtest as B  # noqa: E402
from bikecast import config  # noqa: E402
from bikecast.plots import INK, INK2, MODEL_LABELS, MUTED, PALETTE, SURFACE  # noqa: E402

LAYERS = [
    ("Laptop", "download scripts\ntrips · GBFS ·\nweather · holidays", MUTED),
    ("Volume", "UC Volume\n/Volumes/.../raw\n18 CSV + 3 files", MUTED),
    ("Bronze", "bronze_trips\nall strings +\nsource_file", "#b07a3a"),
    ("Silver", "silver_trips\ndim_stations\ndim_calendar", "#8a8f99"),
    ("Gold", "station_daily\n(zero-filled)\nstation_segments", PALETTE[3]),
    ("Models", "Seasonal Naive\nCroston · TSB\nZIP · Markov", PALETTE[0]),
    ("Serve", "forecasts · metrics\nfuture_forecast\n→ AI/BI dashboard", PALETTE[2]),
]


def architecture(path: pathlib.Path) -> None:
    """Seven-layer architecture diagram with the CI/CD lane underneath."""
    fig, ax = plt.subplots(figsize=(14, 4.6), facecolor=SURFACE)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    w, h, y = 1.7, 1.7, 2.05
    for i, (title, body, color) in enumerate(LAYERS):
        x = 0.25 + i * 1.97
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=color, lw=2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 0.2, body, ha="center", va="center", fontsize=8,
                color=INK2, linespacing=1.4)
        if i < len(LAYERS) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2), (x + 1.97 - 0.02, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))
    ax.text(0.25, 4.35, "Bike-sharing demand forecasting · laptop to dashboard in 7 layers",
            fontsize=13, color=INK, fontweight="bold")
    ax.text(4.19, 4.08, "Databricks (serverless, Unity Catalog, Delta) · one job runs notebooks 01 → 06",
            fontsize=9, color=INK2)
    lane = ("GitHub  ─▶  CI: ruff · pytest (3.11/3.12) · Spark SQL tests · synthetic end-to-end run"
            "  ─▶  CD: databricks bundle deploy  ─▶  job bikeshare_forecast_pipeline")
    ax.add_patch(FancyBboxPatch((0.25, 0.55), 13.5, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#f0efec", ec="#c3c2b7", lw=1))
    ax.text(7.0, 1.05, lane, ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0.25, 0.2, "src/bikecast holds all logic (tested locally and in CI); notebooks only call it.",
            fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _slide(pdf: PdfPages, title: str, bullets: list[str] | None = None, image=None,
           table: pd.DataFrame | None = None) -> None:
    fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
    fig.text(0.05, 0.9, title, fontsize=24, fontweight="bold", color=INK)
    top = 0.8
    if bullets:
        for i, b in enumerate(bullets):
            fig.text(0.06, top - i * 0.065, "•  " + b, fontsize=14, color=INK2, va="top", wrap=True)
        top -= len(bullets) * 0.065 + 0.02
    if image is not None:
        ax = fig.add_axes([0.05, 0.04, 0.9, max(top - 0.06, 0.3)])
        ax.imshow(mpimg.imread(image))
        ax.axis("off")
    if table is not None:
        ax = fig.add_axes([0.06, 0.08, 0.88, max(top - 0.12, 0.3)])
        ax.axis("off")
        tb = ax.table(cellText=table.values, colLabels=table.columns, loc="upper center",
                      cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(13)
        tb.scale(1, 2.2)
        for (r, _c), cell in tb.get_celld().items():
            cell.set_edgecolor("#e1e0d9")
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#f0efec")
        for r in range(1, len(table) + 1):  # highlight the best model per segment
            vals = [float(v) for v in table.iloc[r - 1, 2:]]
            best = 2 + vals.index(min(vals))
            tb[r, best].set_text_props(fontweight="bold", color="#006300")
            tb[r, best].set_facecolor("#e4f3e4")
    pdf.savefig(fig)
    plt.close(fig)


def slides(path: pathlib.Path, outputs: pathlib.Path, docs: pathlib.Path) -> None:
    """Five slides: problem, data & architecture, segmentation, results, recommendation."""
    info = json.loads((outputs / "results.json").read_text(encoding="utf-8"))
    summary = pd.read_parquet(outputs / "metrics_summary.parquet")
    segs = pd.read_parquet(outputs / "station_segments.parquet")
    g = info["gold"]
    head = B.headline(summary)
    table = B.mase_table(summary)
    models = [m for m in config.MODELS if m in table.columns]
    tbl = table.reset_index()
    tbl = pd.DataFrame({"Segment": tbl.segment, "Stations": tbl.stations.astype(int),
                        **{MODEL_LABELS[m]: tbl[m].map("{:.2f}".format) for m in models}})
    quiet = segs.segment.isin(["intermittent", "lumpy"])
    quiet_zero = 100 * segs.loc[quiet, "pct_zero_days"].mean()
    lumpy, inter = head.get("lumpy", {}), head.get("intermittent", {})

    with PdfPages(path) as pdf:
        _slide(pdf, "1 · Problem: quiet stations break generic forecasts", [
            f"{quiet.sum()} of {len(segs)} Capital Bikeshare stations ({100 * quiet.mean():.0f}%) are "
            f"intermittent or lumpy: on average {quiet_zero:.0f}% of their days have zero trips.",
            "A day-of-week naive forecast copies last week's zeros and spikes, so it is wrong on both.",
            "Goal: predict whether a trip happens and how many, and beat the simple baseline.",
            "Same problem as retail items that sell on only a few days (RetSci Topic 1).",
        ], image=docs / "figures" / "example_series.png")
        _slide(pdf, "2 · Data and architecture", [
            f"Jan 2025 – Jun 2026 trip files: {info['silver_removed']['bronze_rows']:,} raw rows → "
            f"{info['silver_removed']['silver_rows']:,} clean trips.",
            f"Gold: {g['stations']} stations × {g['days_per_station']:.0f} days, "
            f"{100 * g['pct_zero_days']:.1f}% of station-days have zero trips.",
        ], image=docs / "architecture.png")
        _slide(pdf, "3 · Segmentation (ADI / CV², Syntetos & Boylan 2005)", [
            "Computed on training data only (≤ 2 June 2026). "
            + ", ".join(f"{k}: {v}" for k, v in info["segments"].items()),
        ], image=docs / "figures" / "adi_cv2_scatter.png")
        _slide(pdf, "4 · Results: median MASE (test days 3–30 June 2026)", [
            "Lower is better; below 1 beats the in-sample seasonal naive. Best per segment in green.",
            f"Lumpy: {MODEL_LABELS.get(lumpy.get('best_model'), '-')} cuts MASE by "
            f"{lumpy.get('reduction_pct', float('nan')):.0f}% vs seasonal naive; intermittent: "
            f"{MODEL_LABELS.get(inter.get('best_model'), '-')} by "
            f"{inter.get('reduction_pct', float('nan')):.0f}%.",
        ], table=tbl)
        _slide(pdf, "5 · Recommendation and next step", [
            "Never use seasonal naive at quiet stations: every intermittent method beats it.",
            "Smooth / erratic stations: ZIP (weather-aware) or TSB give the lowest MASE.",
            "Intermittent: Croston or Markov + NegBin;  lumpy: Markov + NegBin.",
            "Markov also gives P(≥1 trip) per day, useful for rebalancing decisions.",
            "Watch bias: ZIP over-forecasts June at quiet stations; Croston over-forecasts smooth ones.",
            "Next: weekday/weekend transition matrices, hourly grain, probabilistic scoring.",
        ], image=docs / "figures" / "mase_by_segment.png")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--docs", default="docs")
    a = ap.parse_args()
    docs = pathlib.Path(a.docs)
    docs.mkdir(exist_ok=True)
    architecture(docs / "architecture.png")
    slides(docs / "slides.pdf", pathlib.Path(a.outputs), docs)
    print("wrote", docs / "architecture.png", "and", docs / "slides.pdf")


if __name__ == "__main__":
    main()
