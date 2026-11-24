"""Download the current Capital Bikeshare station list from the public GBFS feed.

Usage:
    python scripts/fetch_stations.py [--out data/raw/external]

Writes ``stations.csv`` with station_id, short_name, name, lat, lon, capacity.
In the trip files, ``start_station_id`` usually matches the GBFS ``short_name``.
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import requests

GBFS_ROOT = "https://gbfs.capitalbikeshare.com/gbfs/2.3/gbfs.json"
COLUMNS = ["station_id", "short_name", "name", "lat", "lon", "capacity"]


def main() -> None:
    """Fetch station_information and save it as CSV."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/external")
    args = ap.parse_args()

    root = requests.get(GBFS_ROOT, timeout=30).json()
    feeds = {f["name"]: f["url"] for f in root["data"]["en"]["feeds"]}
    info = requests.get(feeds["station_information"], timeout=30).json()
    df = pd.json_normalize(info["data"]["stations"])
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stations.csv", index=False)
    print(len(df), "stations")


if __name__ == "__main__":
    main()
