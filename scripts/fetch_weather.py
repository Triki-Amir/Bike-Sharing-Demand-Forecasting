"""Download daily weather (Open-Meteo archive) and US/DC public holidays.

Usage:
    python scripts/fetch_weather.py [--out data/raw/external]

Writes ``weather_daily.csv`` (date, temperature_2m_mean, precipitation_sum) and
``holidays.csv`` (date, holiday_name). Observed weather is used for the test days,
as if the weather forecast were perfect (a stated simplification).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import holidays
import pandas as pd
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from bikecast import config  # noqa: E402

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def main() -> None:
    """Fetch weather and holidays for the study period."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/external")
    args = ap.parse_args()

    params = {
        "latitude": 38.9072,
        "longitude": -77.0369,  # Washington DC
        "start_date": config.START,
        "end_date": config.END,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "America/New_York",
    }
    r = requests.get(ARCHIVE, params=params, timeout=60)
    r.raise_for_status()
    w = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})

    years = list(range(int(config.START[:4]), int(config.END[:4]) + 1))
    us = holidays.country_holidays("US", subdiv="DC", years=years)
    h = pd.DataFrame(
        {"date": [d.isoformat() for d in us], "holiday_name": list(us.values())}
    ).sort_values("date")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w.to_csv(out / "weather_daily.csv", index=False)
    h.to_csv(out / "holidays.csv", index=False)
    print(len(w), "weather days,", len(h), "holidays")


if __name__ == "__main__":
    main()
