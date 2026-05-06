"""Data loaders for the bwb framework.

Loaders here read processed CSV/Parquet files and return tidy ``pandas``
DataFrames with normalised column names. They are deliberately thin so they
remain easy to test and swap (e.g., for synthetic data in unit tests).

Conventions
-----------
* All daily series carry a ``date`` column of dtype ``datetime64[ns]``.
* Per-city merged data follow the schema of
  ``data/extracted_csv/merged_by_city/<City>.csv``:
  ``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``.
* Climatological priors are stored in
  ``data_processed/climatological_normals/climatological_priors.json``.
* Distribution atlas lives at
  ``data_processed/distributions/distribution_atlas.{json,parquet}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from bwb.utils.paths import (
    project_data_dir,
    project_processed_dir,
)


# ---------------------------------------------------------------------------
# Daily climate series
# ---------------------------------------------------------------------------


def load_balsas_historical(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the curated Balsas (MA) daily climate series (1961–2025).

    Returns a DataFrame with columns ``date``, ``ETo``, ``pr``.
    """
    if path is None:
        path = project_data_dir("Balsas_MA.csv")
    df = pd.read_csv(path)
    if "Data" in df.columns:
        df = df.rename(columns={"Data": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_city_series(
    city: str,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load merged daily series for a MATOPIBA city.

    Parameters
    ----------
    city : str
        City name matching the CSV stem (e.g., ``"Balsas"``,
        ``"Luis_Eduardo_Magalhaes"``).
    path : Path, optional
        Override path to the CSV file.
    """
    if path is None:
        path = project_data_dir("extracted_csv", "merged_by_city", f"{city}.csv")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Data" in df.columns:
        df = df.rename(columns={"Data": "date"})
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"No 'date' column in {path}")
    return df.sort_values("date").reset_index(drop=True)


def list_available_cities(
    base: Optional[Path] = None,
) -> list[str]:
    """List available city CSV files (stem names)."""
    if base is None:
        base = project_data_dir("extracted_csv", "merged_by_city")
    if not Path(base).exists():
        return []
    return sorted(p.stem for p in Path(base).glob("*.csv"))


def extract_crop_cycle(
    df: pd.DataFrame,
    cycle_start_year: int,
    planting_month: int = 12,
    planting_day: int = 1,
    cycle_days: int = 90,
) -> pd.DataFrame:
    """Extract a single crop-cycle window from a daily climate series.

    Returns a DataFrame indexed 0..cycle_days-1 with a ``dia_ciclo`` column
    (1-based) and the same columns as the input.
    """
    inicio = pd.Timestamp(year=cycle_start_year, month=planting_month, day=planting_day)
    fim = inicio + pd.Timedelta(days=cycle_days - 1)
    mask = (df["date"] >= inicio) & (df["date"] <= fim)
    safra = df.loc[mask].copy()
    if len(safra) != cycle_days:
        raise ValueError(
            f"Cycle {cycle_start_year}/{cycle_start_year + 1} incomplete: "
            f"got {len(safra)} days, expected {cycle_days}"
        )
    safra = safra.reset_index(drop=True)
    safra["dia_ciclo"] = np.arange(1, cycle_days + 1)
    safra["ano_safra"] = cycle_start_year
    return safra


def extract_all_cycles(
    df: pd.DataFrame,
    planting_month: int = 12,
    planting_day: int = 1,
    cycle_days: int = 90,
) -> pd.DataFrame:
    """Extract every complete crop-cycle window in `df`."""
    years = sorted(df["date"].dt.year.unique())
    rows = []
    for year in years:
        try:
            rows.append(extract_crop_cycle(df, year, planting_month, planting_day, cycle_days))
        except ValueError:
            continue
    if not rows:
        raise ValueError("No complete crop cycle found in the input series.")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Oceanic indices
# ---------------------------------------------------------------------------


def load_oceanic_indices(
    indices: Optional[Iterable[str]] = None,
    base: Optional[Path] = None,
) -> pd.DataFrame:
    """Load monthly oceanic indices.

    If multiple individual CSVs exist (``oni.csv``, ``mei.csv`` ...), they are
    merged on the ``date`` column. The consolidated ``oceanic_indices.csv`` is
    preferred when present.
    """
    base = Path(base) if base else project_processed_dir("oceanic_processed")
    if not base.exists():
        raise FileNotFoundError(f"Oceanic processed directory missing: {base}")

    consolidated = base / "oceanic_indices.csv"
    if consolidated.exists():
        df = pd.read_csv(consolidated, parse_dates=["date"])
        if indices:
            keep = ["date"] + [c for c in indices if c in df.columns]
            df = df[keep]
        return df

    # Fallback: merge per-index CSVs
    pieces = []
    for csv in sorted(base.glob("*.csv")):
        if indices and csv.stem not in indices:
            continue
        piece = pd.read_csv(csv, parse_dates=["date"])
        keep_cols = [c for c in piece.columns if c == "date" or c == csv.stem]
        if len(keep_cols) >= 2:
            pieces.append(piece[keep_cols])
    if not pieces:
        raise FileNotFoundError(f"No oceanic index CSV in {base}")
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Climatological priors / distribution atlas
# ---------------------------------------------------------------------------


def load_climatological_priors(
    path: Optional[Path] = None,
) -> dict:
    """Load the climatological priors JSON produced by
    `compute_climatological_normals.py`."""
    if path is None:
        path = project_processed_dir(
            "climatological_normals", "climatological_priors.json"
        )
    if not Path(path).exists():
        raise FileNotFoundError(f"Climatological priors not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_distribution_atlas(
    fmt: str = "parquet",
    path: Optional[Path] = None,
) -> pd.DataFrame | dict:
    """Load the empirical distribution atlas.

    ``fmt`` is either ``"parquet"`` (returns DataFrame) or ``"json"``
    (returns dict).
    """
    base = project_processed_dir("distributions")
    if fmt == "parquet":
        p = Path(path) if path else base / "distribution_atlas.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Distribution atlas parquet missing: {p}")
        return pd.read_parquet(p)
    elif fmt == "json":
        p = Path(path) if path else base / "distribution_atlas.json"
        if not p.exists():
            raise FileNotFoundError(f"Distribution atlas json missing: {p}")
        with open(p) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown fmt: {fmt!r}")


# ---------------------------------------------------------------------------
# Xavier reanalysis (NetCDF)
# ---------------------------------------------------------------------------


def load_xavier_netcdf(
    variable: str,
    path: Optional[Path] = None,
):
    """Open a Xavier reanalysis NetCDF file lazily via xarray.

    Returns the xarray dataset (the caller is responsible for ``.close()``).
    Raises if xarray or the file is unavailable.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            "xarray is required to load Xavier NetCDF: pip install xarray netcdf4"
        ) from e

    if path is None:
        from bwb.utils.paths import project_raw_dir
        path = project_raw_dir("xavier", variable)
        # Either a directory of files or a single file
        if Path(path).is_dir():
            files = sorted(Path(path).glob(f"{variable}_*.nc"))
            if not files:
                raise FileNotFoundError(f"No NetCDF for {variable} in {path}")
            return xr.open_mfdataset(files, combine="by_coords")
    return xr.open_dataset(path)
