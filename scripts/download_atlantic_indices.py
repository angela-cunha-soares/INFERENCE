"""Download Atlantic + extra Pacific oceanic indices for MATOPIBA priors.

The Atlantic meridional SST gradient (TNA-TSA, captured by the AMM) is the
dominant driver of ITCZ position over Northeast Brazil. ENSO (Pacific) is
secondary for MATOPIBA rainfall but still informative. This script pulls all
relevant indices from NOAA/PSL and NOAA/CPC, saves the raw text to
``data_raw/oceanic/`` and writes parsed monthly CSVs to
``data_processed/oceanic_processed/``.

Sources
-------
* NOAA/PSL (https://psl.noaa.gov/data/timeseries/) -- standard ASCII format
  with header ``year_start year_end`` and 12 monthly values per row.
* NOAA/CPC (https://www.cpc.ncep.noaa.gov/data/indices/) -- same ASCII format
  for SOI/NAO, slightly different layout for some files.

Indices downloaded
------------------
Atlantic:
    tna   Tropical North Atlantic SST anomaly (5.5N-23.5N, 15W-57.5W)
    tsa   Tropical South Atlantic SST anomaly (0-20S, 10E-30W)
    atl3  Atlantic Nino, 3S-3N, 0-20W
    amm   Atlantic Meridional Mode (SST projection)
    amo   Atlantic Multidecadal Oscillation, unsmoothed
    nao   North Atlantic Oscillation (CPC monthly)

Pacific (extra to ONI/MEI already in repo):
    nino12  Nino 1+2 SST anomaly (10S-0, 90W-80W)
    nino3   Nino 3   SST anomaly (5S-5N, 150W-90W)
    nino34  Nino 3.4 SST anomaly (5S-5N, 170W-120W)
    nino4   Nino 4   SST anomaly (5S-5N, 160E-150W)
    soi     Southern Oscillation Index (Tahiti-Darwin)

Derived index (computed locally after parsing):
    atl_dipole = tna - tsa  (proxy for ITCZ meridional displacement)

Usage
-----
::

    python scripts/download_atlantic_indices.py
    python scripts/download_atlantic_indices.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw" / "oceanic"
OUT_DIR = ROOT / "data_processed" / "oceanic_processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_START = "1961-01-01"
DATE_END = "2025-12-31"

USER_AGENT = "bwb-research/0.1 (matopiba water-balance project)"
TIMEOUT = 60


@dataclass
class IndexSpec:
    name: str               # short id used as filename and column
    url: str                # NOAA URL
    description: str        # human-readable description
    parser: str             # "psl_yearly" | "cpc_table"
    missing: tuple[float, ...] = (-99.99, -99.9, -99.0, -999.0, -999.9, -9999.0)


# ---------------------------------------------------------------------------
# Index registry
# ---------------------------------------------------------------------------

ATLANTIC_INDICES = [
    IndexSpec("tna", "https://psl.noaa.gov/data/correlation/tna.data",
              "Tropical North Atlantic SST anomaly", "psl_yearly"),
    IndexSpec("tsa", "https://psl.noaa.gov/data/correlation/tsa.data",
              "Tropical South Atlantic SST anomaly", "psl_yearly"),
    IndexSpec("atl3", "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/atl3.long.data",
              "Atlantic Nino (equatorial Atlantic SST)", "psl_yearly"),
    IndexSpec("amm", "https://psl.noaa.gov/data/timeseries/monthly/AMM/ammsst.data",
              "Atlantic Meridional Mode (SST projection)", "psl_yearly"),
    IndexSpec("amo", "https://psl.noaa.gov/data/correlation/amon.us.data",
              "Atlantic Multidecadal Oscillation, unsmoothed", "psl_yearly"),
    IndexSpec("nao",
              "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
              "norm.nao.monthly.b5001.current.ascii.table",
              "North Atlantic Oscillation (CPC monthly, 1950-present)", "cpc_table"),
    IndexSpec("nao_20cr",
              "https://www.psl.noaa.gov/data/20thC_Rean/timeseries/monthly/NAO/"
              "nao.20crv2c.long.data",
              "20th Century Reanalysis NAO (1836-2015, long baseline)",
              "psl_yearly"),
]

PACIFIC_EXTRA_INDICES = [
    IndexSpec("nino12", "https://psl.noaa.gov/data/correlation/nina1.anom.data",
              "Nino 1+2 SST anomaly (coastal Peru)", "psl_yearly"),
    IndexSpec("nino3", "https://psl.noaa.gov/data/correlation/nina3.anom.data",
              "Nino 3 SST anomaly (eastern Pacific)", "psl_yearly"),
    IndexSpec("nino34", "https://psl.noaa.gov/data/correlation/nina34.anom.data",
              "Nino 3.4 SST anomaly (canonical ENSO)", "psl_yearly"),
    IndexSpec("nino4", "https://psl.noaa.gov/data/correlation/nina4.anom.data",
              "Nino 4 SST anomaly (central Pacific)", "psl_yearly"),
    IndexSpec("soi",
              "https://www.cpc.ncep.noaa.gov/data/indices/soi",
              "Southern Oscillation Index (Tahiti-Darwin)", "cpc_table"),
]

ALL_INDICES = ATLANTIC_INDICES + PACIFIC_EXTRA_INDICES


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> int:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            content = resp.read()
    except URLError as e:
        raise RuntimeError(f"download failed: {url} -- {e}") from e
    dest.write_bytes(content)
    return len(content)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_psl_yearly(path: Path, missing: tuple[float, ...]) -> pd.DataFrame:
    """NOAA/PSL standard format:
        line 1:    <year_start> <year_end>
        line 2..N: <year> <jan> <feb> ... <dec>
        last lines: text footer (skipped)
    """
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        first = f.readline().split()
        try:
            year_start = int(first[0])
            year_end = int(first[1])
        except (IndexError, ValueError):
            return pd.DataFrame(columns=["date", "year", "month", "value"])
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if not (year_start <= year <= year_end):
                continue
            if len(parts) < 13:
                continue
            try:
                vals = [float(x) for x in parts[1:13]]
            except ValueError:
                continue
            rows.append([year] + vals)
    if not rows:
        return pd.DataFrame(columns=["date", "year", "month", "value"])
    df = pd.DataFrame(rows, columns=["year"] + list(range(1, 13)))
    long = df.melt(id_vars="year", var_name="month", value_name="value")
    long["month"] = long["month"].astype(int)
    long["date"] = pd.to_datetime(dict(year=long.year, month=long.month, day=1))
    for mv in missing:
        long = long[long["value"] != mv]
    long = long.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    return long[["date", "year", "month", "value"]]


def parse_cpc_table(path: Path, missing: tuple[float, ...]) -> pd.DataFrame:
    """CPC monthly table: header row (text), then ``YEAR JAN FEB ... DEC``.

    Used for both NAO (https://www.cpc.ncep.noaa.gov/.../norm.nao.monthly...)
    and SOI (https://www.cpc.ncep.noaa.gov/data/indices/soi). Both files have
    a text header followed by numeric rows starting with a 4-digit year.

    The SOI file ships *two* sections (anomaly then standardized); we keep the
    first section only by stopping when we hit a year already seen.
    """
    rows = []
    seen_years: set[int] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if not (1850 <= year <= 2100):
                continue
            if year in seen_years:
                # second section started -> stop (we want the anomaly section)
                break
            try:
                vals = [float(x) for x in parts[1:13]]
            except ValueError:
                continue
            seen_years.add(year)
            rows.append([year] + vals)
    if not rows:
        return pd.DataFrame(columns=["date", "year", "month", "value"])
    df = pd.DataFrame(rows, columns=["year"] + list(range(1, 13)))
    long = df.melt(id_vars="year", var_name="month", value_name="value")
    long["month"] = long["month"].astype(int)
    long["date"] = pd.to_datetime(dict(year=long.year, month=long.month, day=1))
    for mv in missing:
        long = long[long["value"] != mv]
    long = long.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    return long[["date", "year", "month", "value"]]


PARSERS = {"psl_yearly": parse_psl_yearly, "cpc_table": parse_cpc_table}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_one(spec: IndexSpec, skip_existing: bool) -> pd.Series | None:
    raw_path = RAW_DIR / f"{spec.name}_raw.txt"
    if skip_existing and raw_path.exists() and raw_path.stat().st_size > 0:
        print(f"  [skip] {spec.name:7s}  (cached at {raw_path.relative_to(ROOT)})")
    else:
        try:
            n_bytes = download(spec.url, raw_path)
            print(f"  [ok]   {spec.name:7s}  {n_bytes/1024:5.1f} KB  <- {spec.url}")
        except RuntimeError as e:
            print(f"  [fail] {spec.name:7s}  {e}")
            return None

    df = PARSERS[spec.parser](raw_path, spec.missing)
    if df.empty:
        print(f"         {spec.name:7s}  parser returned 0 rows")
        return None

    # Save per-index CSV
    csv_path = OUT_DIR / f"{spec.name}.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    s = df.set_index("date")["value"]
    s.name = spec.name
    print(f"         {spec.name:7s}  {len(s):4d} months  "
          f"{s.index.min().date()} -> {s.index.max().date()}  "
          f"mean={s.mean():+.3f}  std={s.std():.3f}")
    return s


def consolidate(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Align all series on a monthly grid and compute Atlantic dipole."""
    full_idx = pd.date_range(DATE_START, DATE_END, freq="MS")
    df = pd.DataFrame(index=full_idx)
    df.index.name = "date"
    for name, s in series.items():
        # protect against duplicate timestamps from upstream files
        if not s.index.is_unique:
            s = s.groupby(level=0).first()
        df[name] = s.reindex(full_idx)

    # Derived: Atlantic dipole (TNA-TSA proxy for ITCZ displacement)
    if "tna" in df.columns and "tsa" in df.columns:
        df["atl_dipole"] = df["tna"] - df["tsa"]

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse raw files already in data_raw/oceanic/")
    args = parser.parse_args()

    print("=" * 78)
    print(" Atlantic + Pacific oceanic indices  ->  data_raw/oceanic/")
    print("=" * 78)
    print(f"Raw : {RAW_DIR.relative_to(ROOT)}")
    print(f"Out : {OUT_DIR.relative_to(ROOT)}")
    print(f"Window: {DATE_START} -> {DATE_END}\n")

    print("[1] Atlantic basin")
    series_dict: dict[str, pd.Series] = {}
    for spec in ATLANTIC_INDICES:
        s = process_one(spec, args.skip_existing)
        if s is not None:
            series_dict[spec.name] = s

    print("\n[2] Pacific basin (extras)")
    for spec in PACIFIC_EXTRA_INDICES:
        s = process_one(spec, args.skip_existing)
        if s is not None:
            series_dict[spec.name] = s

    if not series_dict:
        print("\nNothing downloaded successfully.", file=sys.stderr)
        sys.exit(1)

    print("\n[3] Consolidating to monthly grid + computing derived indices")
    df = consolidate(series_dict)

    out_csv = OUT_DIR / "atlantic_pacific_indices.csv"
    df.to_csv(out_csv, float_format="%.4f")
    print(f"  CSV     : {out_csv.relative_to(ROOT)}  "
          f"({len(df)} months x {len(df.columns)} indices)")

    try:
        out_parquet = OUT_DIR / "atlantic_pacific_indices.parquet"
        df.reset_index().to_parquet(out_parquet, index=False)
        print(f"  Parquet : {out_parquet.relative_to(ROOT)}")
    except ImportError:
        print("  Parquet : skipped (install pyarrow)")

    print("\n[4] Coverage summary")
    n_total = len(df)
    for col in df.columns:
        n_valid = int(df[col].notna().sum())
        cov = 100 * n_valid / n_total
        print(f"  {col:11s}: {n_valid:4d}/{n_total} months ({cov:5.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
