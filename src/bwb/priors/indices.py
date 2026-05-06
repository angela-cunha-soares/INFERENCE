"""Climatic indices used as priors / covariates.

Implements:
* SPEI (Standardized Precipitation-Evapotranspiration Index, Vicente-Serrano
  et al. 2010) - non-parametric Gaussian-CDF version following the IPCC AR6
  recommendation (avoids fitting the log-logistic distribution at every cell).
* IA (Aridity Index = P / ETo, UNEP 1992).
* Quantile-based wet/normal/dry classification (Pinkayan 1966; Xavier et al.
  2002) over the 1991-2020 WMO normal.

All functions accept aligned ``pandas`` Series or NumPy arrays. NaN values
propagate through standardisation but are not dropped.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Aridity Index (UNEP 1992)
# ---------------------------------------------------------------------------


def aridity_index(
    precipitation: np.ndarray | pd.Series,
    eto: np.ndarray | pd.Series,
) -> np.ndarray:
    """Aridity Index = P / ETo.

    UNEP (1992) classes (annual aggregates):
        IA < 0.05      : Hyperarid
        0.05 <= IA < 0.20  : Arid
        0.20 <= IA < 0.50  : Semi-arid
        0.50 <= IA < 0.65  : Dry sub-humid
        IA >= 0.65    : Humid

    Returns the IA as a 1-D ``np.ndarray`` of the same length as the inputs.
    Division by zero (or near-zero ETo) yields NaN.
    """
    p = np.asarray(precipitation, dtype=float)
    e = np.asarray(eto, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = e > 1e-9
    out[mask] = p[mask] / e[mask]
    return out


def classify_aridity(ia: np.ndarray | pd.Series) -> np.ndarray:
    """UNEP 1992 aridity classes."""
    ia_arr = np.asarray(ia, dtype=float)
    classes = np.full(ia_arr.shape, "unknown", dtype=object)
    classes[ia_arr < 0.05] = "hyperarid"
    classes[(ia_arr >= 0.05) & (ia_arr < 0.20)] = "arid"
    classes[(ia_arr >= 0.20) & (ia_arr < 0.50)] = "semi_arid"
    classes[(ia_arr >= 0.50) & (ia_arr < 0.65)] = "dry_subhumid"
    classes[ia_arr >= 0.65] = "humid"
    return classes


# ---------------------------------------------------------------------------
# SPEI (Vicente-Serrano et al. 2010)
# ---------------------------------------------------------------------------


def water_balance_d(
    precipitation: np.ndarray | pd.Series,
    eto: np.ndarray | pd.Series,
) -> np.ndarray:
    """Climatic water balance D = P - ETo (mm)."""
    return np.asarray(precipitation, dtype=float) - np.asarray(eto, dtype=float)


def standardise_nonparametric(values: np.ndarray) -> np.ndarray:
    """Standardise a 1-D series via empirical CDF + inverse normal CDF.

    Used by the IPCC AR6 SPEI definition: avoids fitting the log-logistic
    distribution to short series and is more robust to outliers.

    Returns the standardised series with the same shape (NaN preserved).
    """
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if finite.sum() < 3:
        return out

    finite_vals = arr[finite]
    n = finite_vals.size
    # Use mid-rank empirical CDF in (0, 1) to avoid +/- inf
    ranks = pd.Series(finite_vals).rank(method="average").to_numpy()
    p = (ranks - 0.5) / n
    out[finite] = norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def spei(
    precipitation: np.ndarray | pd.Series,
    eto: np.ndarray | pd.Series,
    *,
    timescale: int = 1,
    by_month: Optional[np.ndarray | pd.Series] = None,
) -> np.ndarray:
    """Standardised Precipitation-Evapotranspiration Index.

    Parameters
    ----------
    precipitation, eto : array-like
        Monthly (or daily aggregated to month) totals of P and ETo (mm).
    timescale : int
        SPEI accumulation window in periods (months), e.g. 1, 3, 6, 12.
    by_month : array-like, optional
        Calendar month of each entry (1-12). When provided, the
        standardisation is performed per-month so seasonality is removed
        following Vicente-Serrano et al. 2010.

    Returns
    -------
    np.ndarray
        SPEI series of the same length as the inputs (the first
        ``timescale - 1`` entries are NaN).
    """
    d = water_balance_d(precipitation, eto)
    if timescale > 1:
        d_acc = pd.Series(d).rolling(window=timescale, min_periods=timescale).sum().to_numpy()
    else:
        d_acc = d

    if by_month is None:
        return standardise_nonparametric(d_acc)

    months = np.asarray(by_month).astype(int)
    out = np.full(d_acc.shape, np.nan, dtype=float)
    for m in range(1, 13):
        mask = months == m
        if mask.sum() >= 3:
            out[mask] = standardise_nonparametric(d_acc[mask])
    return out


def classify_spei(values: np.ndarray | pd.Series) -> np.ndarray:
    """McKee/Vicente-Serrano severity classes for SPEI/SPI.

        SPEI <= -2.0       : extremely dry
        -2.0 <  SPEI <= -1.5 : severely dry
        -1.5 <  SPEI <= -1.0 : moderately dry
        -1.0 <  SPEI <  +1.0 : near normal
        +1.0 <= SPEI <  +1.5 : moderately wet
        +1.5 <= SPEI <  +2.0 : severely wet
        SPEI >= +2.0       : extremely wet
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, "unknown", dtype=object)
    out[v <= -2.0] = "extremely_dry"
    out[(v > -2.0) & (v <= -1.5)] = "severely_dry"
    out[(v > -1.5) & (v <= -1.0)] = "moderately_dry"
    out[(v > -1.0) & (v < 1.0)] = "near_normal"
    out[(v >= 1.0) & (v < 1.5)] = "moderately_wet"
    out[(v >= 1.5) & (v < 2.0)] = "severely_wet"
    out[v >= 2.0] = "extremely_wet"
    return out


# ---------------------------------------------------------------------------
# Quantile-based classification (Pinkayan 1966; Xavier et al. 2002)
# ---------------------------------------------------------------------------


def quantile_classify(
    series: np.ndarray | pd.Series,
    *,
    reference: Optional[np.ndarray | pd.Series] = None,
    low: float = 0.33,
    high: float = 0.67,
) -> np.ndarray:
    """Classify each value as dry / normal / wet using empirical quantiles.

    The defaults follow the tercile convention used in MATOPIBA studies:
    quantile <= 0.33 -> "dry", quantile >= 0.67 -> "wet", else "normal".

    Parameters
    ----------
    series : array-like
        Values to classify.
    reference : array-like, optional
        Reference period from which the quantile thresholds are computed.
        If None, uses the series itself (in-sample classification).
    low, high : float
        Lower and upper tercile probabilities.
    """
    s = np.asarray(series, dtype=float)
    ref = np.asarray(reference if reference is not None else series, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(s.shape, "unknown", dtype=object)
    q_lo = float(np.quantile(ref, low))
    q_hi = float(np.quantile(ref, high))

    out = np.full(s.shape, "normal", dtype=object)
    out[s <= q_lo] = "dry"
    out[s >= q_hi] = "wet"
    out[~np.isfinite(s)] = "unknown"
    return out


# ---------------------------------------------------------------------------
# Convenience: per-cycle scalar indices for a season
# ---------------------------------------------------------------------------


def season_summary(
    precipitation: np.ndarray | pd.Series,
    eto: np.ndarray | pd.Series,
) -> dict:
    """Return seasonal scalar indices for a single growing cycle."""
    p = np.asarray(precipitation, dtype=float)
    e = np.asarray(eto, dtype=float)
    p_total = float(np.nansum(p))
    eto_total = float(np.nansum(e))
    return {
        "P_total_mm": p_total,
        "ETo_total_mm": eto_total,
        "deficit_mm": eto_total - p_total,
        "aridity_index": p_total / eto_total if eto_total > 0 else float("nan"),
        "n_dry_days": int(np.sum(p < 1.0)),
    }
