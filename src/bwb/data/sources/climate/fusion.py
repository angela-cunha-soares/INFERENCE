"""Multi-source climate data fusion.

Adapted from EVAOnline (``backend/core/data_processing/climate_fusion.py``).
Implements per-variable weighted averaging of NASA POWER, Open-Meteo
Archive, and Open-Meteo Forecast outputs, using weights calibrated by the
EVAOnline team against BR-DWGD at 17 Brazilian sites.

Design choices for the bwb port
-------------------------------
* **No Kalman filter.** EVAOnline's Kalman post-processing depends on
  monthly normals from a curated list of cities (``info_cities.csv``)
  that we do not maintain inside INFERENCE. Removing it cuts a heavy
  dependency on a private climatology table while preserving the part
  of the pipeline that has the largest documented effect on skill —
  the per-variable source weighting.
* **No region detection.** This codebase targets MATOPIBA / Brazil.
  EVAOnline's ``GLOBAL`` / ``USA`` / ``NORDIC`` branches are removed.
* **ETo is recomputed from fused inputs.** Following ``eto_services.py``
  in EVAOnline, after fusing the raw drivers (Tmax, Tmin, RH, u2, Rs)
  we call :func:`bwb.data.sources.climate.nasa_power.compute_eto_fao56_pm`
  on the fused series. This is more rigorous than averaging two
  independently-computed ETo time series.

Usage
-----
>>> from datetime import date
>>> from bwb.data.sources.climate import (
...     download_nasa_power, download_openmeteo_archive,
...     download_openmeteo_forecast, fuse_climate_sources,
... )
>>> nasa = download_nasa_power(lat, lon, start, end, elevation_m=285.0)
>>> archive = download_openmeteo_archive(lat, lon, start, end)
>>> fused = fuse_climate_sources(
...     {"nasa_power": nasa, "openmeteo_archive": archive},
...     lat=lat, elevation_m=285.0,
... )

Reference
---------
EVAOnline ``HIST_WEIGHTS`` (calibrated 2025 against BR-DWGD).
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from bwb.data.sources.climate.nasa_power import compute_eto_fao56_pm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-variable NASA-POWER weight (Open-Meteo gets 1 - w)
# ---------------------------------------------------------------------------

# These are the EVAOnline ``HIST_WEIGHTS`` from
# ``climate_fusion.py``. Comments quote EVAOnline's notes on which
# product was found to perform best.
HIST_WEIGHTS_NASA: Dict[str, float] = {
    "Tmax": 0.58,   # NASA slight advantage in extremes
    "Tmin": 0.52,   # near-equal performance
    "RH":   0.35,   # ERA5 humidity superior
    "u2":   0.20,   # ERA5 wind more accurate
    "Rs":   0.92,   # CERES satellite radiation dominant
    "pr":   0.50,   # equal reliability; per-source QC handles outliers
}

# Variables we fuse (in this order). ``ETo`` is intentionally NOT in the
# list — it is recomputed from the fused inputs at the end.
FUSED_VARS = ("Tmax", "Tmin", "RH", "u2", "Rs", "pr")

# Recognised source names. Any source not in this set gets equal weight
# with whichever NASA-or-OM weight is closest semantically.
NASA_LIKE = {"nasa_power"}
OPENMETEO_LIKE = {"openmeteo_archive", "openmeteo_forecast"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fuse_climate_sources(
    sources: Mapping[str, pd.DataFrame],
    *,
    lat: float,
    elevation_m: Optional[float] = None,
    recompute_eto: bool = True,
) -> pd.DataFrame:
    """Fuse a set of climate-source DataFrames into a single bwb-schema frame.

    Parameters
    ----------
    sources : mapping of source name → DataFrame
        Each DataFrame must follow the bwb canonical schema
        (``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``). Recognised source
        names are ``"nasa_power"``, ``"openmeteo_archive"`` and
        ``"openmeteo_forecast"`` — others fall back to equal weighting.
    lat : float
        Latitude (degrees) — used to recompute ETo from fused inputs.
    elevation_m : float, optional
        Elevation (m above MSL). Required when ``recompute_eto=True``;
        if omitted, the fused ETo is filled by averaging the per-source
        ETo columns instead.
    recompute_eto : bool
        If True (default), call FAO-56 Penman-Monteith on the fused
        Tmax/Tmin/RH/u2/Rs to produce ``ETo``. If False, fuse the
        per-source ETo columns directly with NASA-vs-OM weights.

    Returns
    -------
    DataFrame with the canonical bwb schema, indexed by the union of
    dates available across sources, dropping rows where every source
    is missing for the entire row.
    """
    if not sources:
        raise ValueError("fuse_climate_sources: no sources provided")

    aligned = _align_sources(sources)
    fused = aligned[["date"]].copy()
    weights = _resolve_weights(list(sources.keys()))

    for var in FUSED_VARS:
        fused[var] = _per_row_weighted_mean(aligned, var, weights)

    if recompute_eto:
        if elevation_m is None:
            logger.warning(
                "fuse_climate_sources: elevation_m missing; falling back to "
                "averaging the per-source ETo columns."
            )
            fused["ETo"] = _per_row_weighted_mean(aligned, "ETo", weights)
        else:
            doy = fused["date"].dt.dayofyear.to_numpy()
            Tmax = fused["Tmax"].to_numpy()
            Tmin = fused["Tmin"].to_numpy()
            Tmean = 0.5 * (Tmax + Tmin)
            fused["ETo"] = compute_eto_fao56_pm(
                Tmax=Tmax, Tmin=Tmin, Tmean=Tmean,
                RH=fused["RH"].to_numpy(), u2=fused["u2"].to_numpy(),
                Rs=fused["Rs"].to_numpy(), doy=doy,
                lat_deg=lat, elevation_m=float(elevation_m),
            )
    else:
        fused["ETo"] = _per_row_weighted_mean(aligned, "ETo", weights)

    cols = ["date", "Rs", "u2", "Tmax", "Tmin", "RH", "pr", "ETo"]
    fused = fused[cols].copy()
    # Drop rows where every numeric column is NaN (no source covered this date).
    fused = fused.dropna(subset=cols[1:], how="all").reset_index(drop=True)
    return fused


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _align_sources(sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all source DataFrames on ``date``.

    Returns a wide frame where the value column for ``var`` from source
    ``s`` lives at column ``f"{var}__{s}"``. Sources are not modified.
    """
    pieces = []
    for name, df in sources.items():
        if "date" not in df.columns:
            raise ValueError(f"source {name!r} missing 'date' column")
        sub = df.copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
        sub = sub.sort_values("date").drop_duplicates("date", keep="last")
        rename = {c: f"{c}__{name}" for c in sub.columns if c != "date"}
        pieces.append(sub.rename(columns=rename))

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def _resolve_weights(source_names: list) -> Dict[str, Dict[str, float]]:
    """Map ``var → {source → weight}`` honouring NASA-vs-OM HIST_WEIGHTS.

    For each variable, NASA-like sources share ``HIST_WEIGHTS_NASA[var]``
    equally, OM-like sources share ``1 - HIST_WEIGHTS_NASA[var]`` equally.
    Sources whose names are unrecognised fall back to equal weight (1/N).
    Weights for variables not in ``HIST_WEIGHTS_NASA`` (e.g. ``ETo``)
    default to 0.5/0.5 NASA-vs-OM.
    """
    nasa = [s for s in source_names if s in NASA_LIKE]
    om = [s for s in source_names if s in OPENMETEO_LIKE]
    other = [s for s in source_names if s not in NASA_LIKE and s not in OPENMETEO_LIKE]

    out: Dict[str, Dict[str, float]] = {}
    all_vars = set(FUSED_VARS) | {"ETo"}
    for var in all_vars:
        w_nasa_total = HIST_WEIGHTS_NASA.get(var, 0.5)
        w_om_total = 1.0 - w_nasa_total
        wmap: Dict[str, float] = {}
        if nasa:
            for s in nasa:
                wmap[s] = w_nasa_total / len(nasa)
        if om:
            for s in om:
                wmap[s] = w_om_total / len(om)
        # If only one family is present, give it the full weight share.
        if not nasa and om:
            for s in om:
                wmap[s] = 1.0 / len(om)
        if nasa and not om:
            for s in nasa:
                wmap[s] = 1.0 / len(nasa)
        # Unrecognised sources: add equal extra share, then renormalise.
        if other:
            extra = 1.0 / max(len(source_names), 1)
            for s in other:
                wmap[s] = extra
            total = sum(wmap.values())
            wmap = {k: v / total for k, v in wmap.items()}
        out[var] = wmap
    return out


def _per_row_weighted_mean(aligned: pd.DataFrame, var: str,
                           weights: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Row-wise weighted mean over the available (non-NaN) sources.

    For each row, the weights of the *available* sources are renormalised
    to sum to 1; rows with no available source produce NaN.
    """
    cols = [c for c in aligned.columns if c.startswith(f"{var}__")]
    if not cols:
        return np.full(len(aligned), np.nan)
    matrix = aligned[cols].to_numpy(dtype=float)
    src_names = [c.split("__", 1)[1] for c in cols]
    w_full = np.array([weights[var].get(s, 0.0) for s in src_names], dtype=float)
    if w_full.sum() == 0:
        return np.full(matrix.shape[0], np.nan)

    mask = np.isfinite(matrix)
    w_row = mask * w_full
    w_sum = w_row.sum(axis=1)
    out = np.full(matrix.shape[0], np.nan)
    valid = w_sum > 0
    matrix_safe = np.where(mask, matrix, 0.0)
    out[valid] = (matrix_safe[valid] * w_row[valid]).sum(axis=1) / w_sum[valid]
    return out


__all__ = ["fuse_climate_sources", "HIST_WEIGHTS_NASA", "FUSED_VARS"]
