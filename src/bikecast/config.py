"""Project configuration: catalog, schema, paths, dates and thresholds.

Every path, table name and date used by notebooks and the local pipeline lives here.
"""

from __future__ import annotations

# --- Databricks / Unity Catalog -------------------------------------------------
CATALOG = "workspace"  # default catalog in Databricks Free Edition
SCHEMA = "bikeshare"
VOLUME = "raw"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
TRIPS_GLOB = f"{VOLUME_PATH}/trips/*.csv"
EXTERNAL_PATH = f"{VOLUME_PATH}/external"

# --- Local paths (laptop / CI) --------------------------------------------------
LOCAL_DATA_DIR = "data"
LOCAL_RAW_TRIPS = "raw/trips"
LOCAL_RAW_EXTERNAL = "raw/external"
LOCAL_OUTPUT = "outputs"

# --- Study period ---------------------------------------------------------------
START, END = "2025-01-01", "2026-06-30"
N_DAYS = 546  # days between START and END, inclusive

# Stations must exist for the whole period (avoid "new station looks intermittent").
ELIGIBLE_FIRST_TRIP_BY = "2025-01-31"
ELIGIBLE_LAST_TRIP_FROM = "2026-06-01"

# --- Backtest -------------------------------------------------------------------
HORIZON, N_WINDOWS = 7, 4  # 4 rolling windows of 7 days ending on END
TRAIN_END = "2026-06-02"  # first cutoff: segmentation and MASE scale use data <= this date
SEASON_LENGTH = 7

# --- Segmentation (Syntetos & Boylan, 2005) -------------------------------------
ADI_CUT, CV2_CUT = 1.32, 0.49
MIN_NONZERO_DAYS = 10  # fewer active days than this: too little to model

# --- Silver cleaning rules ------------------------------------------------------
MIN_DURATION_S, MAX_DURATION_S = 60, 86_400

# --- Tuning grids (chosen on the first two cutoffs only) ------------------------
TSB_ALPHA_GRID = (0.05, 0.1, 0.2, 0.3)
MARKOV_WINDOW_GRID = (90, 180, 365)
DEFAULT_TSB_ALPHAS = (0.1, 0.1)
DEFAULT_MARKOV_WINDOW = 180

MODELS = ["SeasonalNaive", "CrostonClassic", "TSB", "ZIP", "Markov"]
SEGMENTS = ["smooth", "erratic", "intermittent", "lumpy"]

TABLES = [
    "bronze_trips",
    "silver_trips",
    "dim_stations",
    "dim_calendar",
    "station_daily",
    "station_segments",
    "forecasts",
    "metrics",
    "metrics_summary",
    "future_forecast",
]


def t(name: str, catalog: str = CATALOG, schema: str = SCHEMA) -> str:
    """Return the fully qualified Unity Catalog name of a project table."""
    return f"{catalog}.{schema}.{name}"
