"""Shared fixtures: make src importable and build a small synthetic dataset once per session."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def _synthetic(tmp_path_factory) -> tuple[pathlib.Path, dict]:
    """Generate synthetic raw data once (2 stations per segment + 1 new station)."""
    from bikecast import synthetic

    d = tmp_path_factory.mktemp("synthetic")
    return d, synthetic.generate(d, stations_per_segment=2, seed=7)


@pytest.fixture(scope="session")
def synthetic_dir(_synthetic) -> pathlib.Path:
    """Directory holding raw/trips and raw/external synthetic files."""
    return _synthetic[0]


@pytest.fixture(scope="session")
def manifest(_synthetic) -> dict:
    """Ground truth written by the synthetic generator."""
    return _synthetic[1]


@pytest.fixture(scope="session")
def layers(synthetic_dir) -> dict:
    """Bronze, silver and gold built with the local pandas pipeline on the synthetic data."""
    import pandas as pd

    from bikecast import pipeline
    from bikecast.segmentation import segment_stations

    ext = synthetic_dir / "raw" / "external"
    bronze = pipeline.read_bronze(synthetic_dir / "raw" / "trips")
    silver, removed = pipeline.clean_trips(bronze)
    gbfs = pd.read_csv(ext / "stations.csv", dtype={"short_name": str, "station_id": str})
    dim = pipeline.build_dim_stations(silver, gbfs)
    cal = pipeline.build_calendar(pd.read_csv(ext / "weather_daily.csv"),
                                  pd.read_csv(ext / "holidays.csv"))
    daily = pipeline.build_station_daily(silver, dim, cal)
    segs = segment_stations(daily)
    return {"bronze": bronze, "silver": silver, "removed": removed, "gbfs": gbfs, "dim": dim,
            "cal": cal, "daily": daily, "segs": segs}
