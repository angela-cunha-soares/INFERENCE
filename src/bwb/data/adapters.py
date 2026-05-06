"""Adapters that turn DataFrames / dicts into model-ready inputs.

Most callers want to take a daily-climate DataFrame for a single crop cycle
(``date, ETo, pr`` minimum) and feed it into the Bayesian water balance model.
This module provides:

* :func:`build_water_balance_inputs` — DataFrame → :class:`WaterBalanceInputs`
* :func:`run_deterministic_baseline` — produces the reference theta series
  used as the observation target for the Bayesian model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from bwb.models.water_balance import WaterBalanceInputs
from bwb.phenology.kc_curves import fao56_kc_curve, fao56_kc_5stage_step


# ---------------------------------------------------------------------------
# Deterministic FAO-56 baseline (used as observation surrogate for the Bayes)
# ---------------------------------------------------------------------------


def run_deterministic_baseline(
    p_daily: np.ndarray,
    eto_daily: np.ndarray,
    kc_curve: np.ndarray,
    theta_s: float = 0.45,
    theta_r: float = 0.10,
    theta_init: Optional[float] = None,
    z_r_m: float = 0.60,
    p_eff_factor: float = 0.8,
) -> np.ndarray:
    """Run the FAO-56 daily water-balance with fixed parameters.

    Returns the daily soil-moisture trajectory (volumetric, m³/m³).
    """
    n_days = len(p_daily)
    if theta_init is None:
        theta_init = 0.5 * (theta_s + theta_r)

    z_r_mm = z_r_m * 1000.0
    theta = np.empty(n_days, dtype=float)
    theta[0] = theta_init
    for d in range(1, n_days):
        etc = kc_curve[d] * eto_daily[d]
        balance = theta[d - 1] + (p_eff_factor * p_daily[d] - etc) / z_r_mm
        theta[d] = float(np.clip(balance, theta_r, theta_s))
    return theta


def build_kc_curve(
    profile: dict,
    n_days: Optional[int] = None,
) -> np.ndarray:
    """Build a daily Kc curve from a regional profile dict.

    Dispatches to one of two schemes based on which keys the profile exposes:

    * **5-stage step** (preferred, FAO-56 Cap. 6 simplified, Embrapa convention):
      requires ``kc_ini, kc_dev, kc_mid, kc_late, kc_harvest`` and matching
      ``L_*`` plus ``L_harvest``. Each phase has a constant Kc value.
    * **4-stage linear** (legacy, FAO-56 Table 12 canonical): ``kc_ini, kc_mid,
      kc_end, L_ini..L_late``; Kc ramps linearly between reference points.
    """
    crop = profile.get("crop", profile)

    has_5stage = all(k in crop for k in
                     ("kc_dev", "kc_late", "kc_harvest", "L_harvest"))
    if has_5stage:
        kc = fao56_kc_5stage_step(
            kc_ini=float(crop["kc_ini"]),
            kc_dev=float(crop["kc_dev"]),
            kc_mid=float(crop["kc_mid"]),
            kc_late=float(crop["kc_late"]),
            kc_harvest=float(crop["kc_harvest"]),
            L_ini=int(crop["L_ini"]),
            L_dev=int(crop["L_dev"]),
            L_mid=int(crop["L_mid"]),
            L_late=int(crop["L_late"]),
            L_harvest=int(crop["L_harvest"]),
        )
    else:
        kc = fao56_kc_curve(
            kc_ini=float(crop["kc_ini"]),
            kc_mid=float(crop["kc_mid"]),
            kc_end=float(crop["kc_end"]),
            L_ini=int(crop["L_ini"]),
            L_dev=int(crop["L_dev"]),
            L_mid=int(crop["L_mid"]),
            L_late=int(crop["L_late"]),
        )
    if n_days is not None and len(kc) != n_days:
        if len(kc) > n_days:
            kc = kc[:n_days]
        else:
            kc = np.concatenate([kc, np.full(n_days - len(kc), kc[-1])])
    return kc


def get_awc_for_city(profile: dict, city: str) -> float:
    """Resolve the per-city AWC (mm) from the regional profile.

    Looks up ``profile["soils"][city]["awc_mm"]`` first; falls back to the
    legacy uniform ``profile["crop"]["awc_mm"]`` when the per-city block is
    not present (e.g. for profiles that haven't been migrated yet, or callers
    that don't know which city they're working with).

    Per-city AWC values come from the SNIRH/ANA-UFPR CAD dataset
    (Santos et al. TED ANA-UFPR; SNIRH 28fe4baa-66f3-4f6b-b0d2-890abf5910c4),
    with AWC_mm = CAD × Z_r(60 cm) × 10.
    """
    soils = profile.get("soils") or {}
    city_block = soils.get(city)
    if isinstance(city_block, dict) and "awc_mm" in city_block:
        return float(city_block["awc_mm"])
    crop = profile.get("crop", profile)
    return float(crop["awc_mm"])


# ---------------------------------------------------------------------------
# DataFrame → model inputs
# ---------------------------------------------------------------------------


def build_water_balance_inputs(
    cycle_df: pd.DataFrame,
    profile: dict,
    *,
    city: str = "",
    season: str = "",
    crop_name: str = "soybean",
    p_eff_factor: float = 0.8,
    theta_init: Optional[float] = None,
) -> WaterBalanceInputs:
    """Convert a per-cycle climate DataFrame into :class:`WaterBalanceInputs`.

    Parameters
    ----------
    cycle_df : pd.DataFrame
        Must contain ``ETo`` and ``pr`` columns (mm/day), one row per cycle day.
    profile : dict
        Regional profile (see :func:`bwb.config.profiles.load_profile`).
    city, season, crop_name : str
        Annotation passed through to the model inputs.
    p_eff_factor : float
        Effective precipitation factor (default 0.8 from FAO-56 simplification).
    theta_init : float, optional
        Initial volumetric soil moisture; defaults to mid-range (theta_s+theta_r)/2.
    """
    if "ETo" not in cycle_df.columns or "pr" not in cycle_df.columns:
        raise ValueError("cycle_df must have 'ETo' and 'pr' columns")

    p_daily = cycle_df["pr"].to_numpy(dtype=float)
    eto_daily = cycle_df["ETo"].to_numpy(dtype=float)
    n_days = len(p_daily)

    kc = build_kc_curve(profile, n_days=n_days)

    soil = profile.get("soil", {})
    theta_s = float(soil.get("theta_s", 0.43))
    theta_r = float(soil.get("theta_r", 0.045))

    theta_obs = run_deterministic_baseline(
        p_daily=p_daily,
        eto_daily=eto_daily,
        kc_curve=kc,
        theta_s=theta_s,
        theta_r=theta_r,
        theta_init=theta_init,
        z_r_m=float(profile.get("crop", {}).get("root_depth_cm", 60)) / 100.0,
        p_eff_factor=p_eff_factor,
    )

    return WaterBalanceInputs(
        P_daily=p_daily,
        ETo_daily=eto_daily,
        Kc_curve=kc,
        theta_observed=theta_obs,
        soil_params={"theta_s": theta_s, "theta_r": theta_r},
        Z_r=float(profile.get("crop", {}).get("root_depth_cm", 60)) / 100.0,
        crop_name=crop_name,
        city=city,
        season=season,
    )


def cycle_label(year: int) -> str:
    """Return canonical cycle label like ``"2020/2021"``."""
    return f"{year}/{year + 1}"
