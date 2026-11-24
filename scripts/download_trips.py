"""Download and unzip the Capital Bikeshare monthly trip files (Jan 2025 - Jun 2026).

Usage:
    python scripts/download_trips.py [--out data/raw/trips] [--force]

Each month is saved as ``<out>/<YYYYMM>.csv``. Months that already exist are skipped
unless ``--force`` is given. Spark cannot read zips directly, which is why we unzip here.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys
import time
import zipfile

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from bikecast import config  # noqa: E402

BASE = "https://s3.amazonaws.com/capitalbikeshare-data/{ym}-capitalbikeshare-tripdata.zip"


def months_between(start: str, end: str) -> list[str]:
    """Return YYYYMM strings for every month between two ISO dates (inclusive)."""
    y, m = int(start[:4]), int(start[5:7])
    y_end, m_end = int(end[:4]), int(end[5:7])
    out = []
    while (y, m) <= (y_end, m_end):
        out.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def download_month(ym: str, out: pathlib.Path, retries: int = 3) -> pathlib.Path:
    """Download one monthly zip and write its CSV to ``out/<ym>.csv``."""
    url = BASE.format(ym=ym)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csvs = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        if not csvs:
            raise RuntimeError(f"No CSV found in {url}")
        target = out / f"{ym}.csv"
        # Some months are split in several CSVs: concatenate them, keeping one header.
        with open(target, "wb") as fh:
            for i, name in enumerate(sorted(csvs)):
                data = z.read(name)
                if i > 0:
                    data = data.split(b"\n", 1)[1] if b"\n" in data else b""
                if not data.endswith(b"\n"):
                    data += b"\n"
                fh.write(data)
    return target


def main() -> None:
    """Download all months of the study period."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/trips")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ym in months_between(config.START, config.END):
        target = out / f"{ym}.csv"
        if target.exists() and target.stat().st_size > 0 and not args.force:
            print("skip", ym)
            continue
        path = download_month(ym, out)
        print("ok", ym, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
