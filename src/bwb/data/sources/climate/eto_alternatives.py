"""Alternative reference-ET formulas for sensitivity analysis.

Two reduced-input alternatives to FAO-56 Penman-Monteith are commonly cited
when the full set of climate variables is unavailable (Allen et al. 1998,
Sect. 6 sidebar; Ribeiro et al. 2023, Eqs. 2-3):

* **Hargreaves-Samani (1985)** — needs only Tmax, Tmin and Ra
  (extraterrestrial radiation, derivable from latitude and DOY).
* **Benavides-Lopez (1970)** — needs only Tmean and RH; calibrated for
  the 15°N–15°S tropical band.

These exist in the bwb pipeline for one purpose: to demonstrate, in the
paper, that the **fused NASA + Open-Meteo** product reproduces full
FAO-56 Penman-Monteith more faithfully than either reduced formula
manages to. They are *not* used to compute ETo for the rolling forecast.

Reference
---------
Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop
evapotranspiration: guidelines for computing crop water requirements
(FAO Irrigation and Drainage Paper 56).

Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration
from temperature. *Applied Engineering in Agriculture* 1(2), 96-99.

Benavides, J. G., & Lopez, D. (1970). Formula para el calculo de la
evapotranspiracion potencial adaptada al tropico (15°N-15°S).
*Agronomia Tropical* 20, 335-345.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def _extraterrestrial_radiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """FAO-56 Eq. 21 — Ra (MJ m⁻² day⁻¹) from DOY and latitude (degrees)."""
    lat = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta_sun = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    cos_arg = np.clip(-np.tan(lat) * np.tan(delta_sun), -1.0, 1.0)
    omega_s = np.arccos(cos_arg)
    Gsc = 0.0820  # MJ m⁻² min⁻¹
    Ra = (24.0 * 60.0 / np.pi) * Gsc * dr * (
        omega_s * np.sin(lat) * np.sin(delta_sun)
        + np.cos(lat) * np.cos(delta_sun) * np.sin(omega_s)
    )
    return np.maximum(Ra, 0.0)


def compute_eto_hargreaves_samani(
    Tmax: Union[np.ndarray, float],
    Tmin: Union[np.ndarray, float],
    *,
    doy: np.ndarray,
    lat_deg: float,
) -> np.ndarray:
    """Hargreaves-Samani (1985) reference ET (mm day⁻¹).

    .. math::

       \\mathrm{ET}_{0,\\mathrm{HS}} = 0.0023 \\cdot 0.408 \\cdot R_a \\cdot
                                       (T_\\mathrm{mean} + 17.8) \\cdot
                                       \\sqrt{T_\\mathrm{max} - T_\\mathrm{min}}

    where Ra is extraterrestrial radiation (MJ m⁻² day⁻¹) from FAO-56 Eq. 21
    and the 0.408 factor converts MJ m⁻² day⁻¹ → mm day⁻¹.

    Parameters
    ----------
    Tmax, Tmin : array of °C
    doy : array of day-of-year (1..366)
    lat_deg : decimal degrees

    Returns
    -------
    array (mm day⁻¹) — clipped at 0.
    """
    Tmax = np.asarray(Tmax, dtype=float)
    Tmin = np.asarray(Tmin, dtype=float)
    Tmean = 0.5 * (Tmax + Tmin)
    Ra = _extraterrestrial_radiation(np.asarray(doy), float(lat_deg))
    eto = 0.0023 * 0.408 * Ra * (Tmean + 17.8) * np.sqrt(np.maximum(Tmax - Tmin, 0.0))
    return np.maximum(eto, 0.0)


def compute_eto_benavides_lopez(
    Tmean: Union[np.ndarray, float],
    RHmean: Union[np.ndarray, float],
) -> np.ndarray:
    """Benavides-Lopez (1970) reference ET (mm day⁻¹) — tropical 15°N-15°S.

    .. math::

       \\mathrm{ET}_{0,\\mathrm{BL}} = 1.21 \\cdot 10^{\\dfrac{7.42\\,T_\\mathrm{mean}}
                                                {234.7 + T_\\mathrm{mean}}}
                                  \\cdot (1 - 0.01\\,\\mathrm{RH}_\\mathrm{mean})
                                  + 0.21\\,T_\\mathrm{mean} - 2.30

    Parameters
    ----------
    Tmean : °C
    RHmean : %

    Returns
    -------
    array (mm day⁻¹) — clipped at 0.
    """
    T = np.asarray(Tmean, dtype=float)
    RH = np.asarray(RHmean, dtype=float)
    eto = 1.21 * 10.0 ** (7.42 * T / (234.7 + T)) * (1.0 - 0.01 * RH) \
          + 0.21 * T - 2.30
    return np.maximum(eto, 0.0)


__all__ = ["compute_eto_hargreaves_samani", "compute_eto_benavides_lopez"]
