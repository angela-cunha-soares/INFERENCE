"""FAO-56 daily water balance in millimetres.

This module provides the simple soil-water budget used by the climatological
forecast pipeline (:mod:`bwb.forecast.climatological`). Unlike
:func:`bwb.data.adapters.run_deterministic_baseline` -- which works in
volumetric soil moisture (theta, m^3/m^3) -- this module tracks the soil
water content in mm relative to the AWC (FAO-56 Eq. 84).

State variable
--------------
SW_d : soil water content on day d, mm, in [0, AWC]

Forcings
--------
P_d        : precipitation, mm/day
ETo_d      : reference evapotranspiration, mm/day
Kc_d       : crop coefficient, dimensionless
ETc_d      : Kc_d x ETo_d  (crop evapotranspiration)
P_eff_d    : p_eff_factor x P_d  (effective precipitation, FAO-56 simplified)
MAD        : management allowed depletion (fraction of AWC)
threshold  : AWC * (1 - MAD)

Daily update
------------
SW_new = SW_prev + P_eff - ETc
DP     = max(0, SW_new - AWC)               # deep percolation
SW_new = min(SW_new, AWC)
if SW_new < threshold:
    I  = AWC - SW_new                        # irrigation refill to FC
    SW_new = AWC
SW_new = max(0, SW_new)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WaterBalanceResult:
    """Daily output of the FAO-56 water balance."""

    SW: np.ndarray      # soil water content (mm), shape (n_days,)
    I: np.ndarray       # irrigation depth (mm/day), shape (n_days,)
    DP: np.ndarray      # deep percolation (mm/day), shape (n_days,)
    ETc: np.ndarray     # crop evapotranspiration (mm/day), shape (n_days,)


def daily_water_balance(
    eto: np.ndarray,
    pr: np.ndarray,
    kc: np.ndarray,
    *,
    awc: float,
    mad: float = 0.55,
    p_eff_factor: float = 0.8,
    sw_init: float | None = None,
) -> WaterBalanceResult:
    """Run the FAO-56 daily water balance for a single crop cycle.

    Parameters
    ----------
    eto, pr, kc : array-like (n_days,)
        Daily reference evapotranspiration (mm), precipitation (mm) and
        crop coefficient.
    awc : float
        Available water capacity in mm (e.g. 120 mm for a 60 cm root zone in
        sandy loam).
    mad : float
        Management allowed depletion as a fraction of AWC (default 0.55,
        FAO-56 default for soybean).
    p_eff_factor : float
        Effective precipitation factor (default 0.8, FAO-56 simplified).
    sw_init : float, optional
        Initial soil water content in mm. Defaults to ``awc`` (full at
        planting), which mirrors the legacy ``temp/main.py`` behaviour.

    Returns
    -------
    WaterBalanceResult
    """
    eto = np.asarray(eto, dtype=float)
    pr = np.asarray(pr, dtype=float)
    kc = np.asarray(kc, dtype=float)

    n = len(eto)
    if not (len(pr) == n == len(kc)):
        raise ValueError(
            f"length mismatch: eto={len(eto)}, pr={len(pr)}, kc={len(kc)}"
        )

    etc = kc * eto
    p_eff = p_eff_factor * pr

    SW = np.zeros(n, dtype=float)
    I = np.zeros(n, dtype=float)
    DP = np.zeros(n, dtype=float)

    sw = float(sw_init) if sw_init is not None else float(awc)
    threshold = awc * (1.0 - mad)

    for d in range(n):
        sw_new = sw + p_eff[d] - etc[d]
        dp = max(0.0, sw_new - awc)
        sw_new = min(sw_new, awc)
        irrig = 0.0
        if sw_new < threshold:
            irrig = awc - sw_new
            sw_new = awc
        sw_new = max(0.0, sw_new)
        SW[d] = sw_new
        I[d] = irrig
        DP[d] = dp
        sw = sw_new

    return WaterBalanceResult(SW=SW, I=I, DP=DP, ETc=etc)
