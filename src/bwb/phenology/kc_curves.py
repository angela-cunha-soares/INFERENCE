"""FAO-56 crop coefficient (Kc) curves following Eq. 66 (Allen et al. 1998).

Implements piecewise-linear interpolation between the four canonical
phenological stages: initial, development, mid-season, late-season.

Reference values for soybean (FAO-56 Table 12):
    Kc_ini = 0.40,  Kc_mid = 1.15,  Kc_end = 0.50

Stage durations for early-cycle Brazilian soybean (90-day total):
    L_ini=15, L_dev=15, L_mid=40, L_late=20
"""

from __future__ import annotations

import numpy as np


def fao56_kc_curve(
    kc_ini: float,
    kc_mid: float,
    kc_end: float,
    L_ini: int,
    L_dev: int,
    L_mid: int,
    L_late: int,
) -> np.ndarray:
    """Generate daily Kc series following FAO-56 Eq. 66.

    Stages:
        initial     : constant Kc_ini for L_ini days
        development : linear ramp Kc_ini → Kc_mid over L_dev days
        mid-season  : constant Kc_mid for L_mid days
        late-season : linear ramp Kc_mid → Kc_end over L_late days

    Parameters
    ----------
    kc_ini, kc_mid, kc_end : float
        Crop coefficients at the three FAO-56 reference points.
    L_ini, L_dev, L_mid, L_late : int
        Stage durations in days.

    Returns
    -------
    np.ndarray
        Vector of length `L_ini + L_dev + L_mid + L_late`.
    """
    if min(L_ini, L_dev, L_mid, L_late) < 0:
        raise ValueError("Stage durations must be non-negative.")
    n = L_ini + L_dev + L_mid + L_late
    if n == 0:
        return np.array([], dtype=float)

    kc = np.empty(n, dtype=float)

    # 1) Initial — constant
    kc[:L_ini] = kc_ini

    # 2) Development — linear ramp ini → mid
    if L_dev > 0:
        f = np.arange(1, L_dev + 1) / L_dev
        kc[L_ini : L_ini + L_dev] = kc_ini + f * (kc_mid - kc_ini)

    # 3) Mid-season — constant
    if L_mid > 0:
        s = L_ini + L_dev
        kc[s : s + L_mid] = kc_mid

    # 4) Late-season — linear ramp mid → end
    if L_late > 0:
        f = np.arange(1, L_late + 1) / L_late
        kc[L_ini + L_dev + L_mid :] = kc_mid + f * (kc_end - kc_mid)

    return kc


def soybean_kc_90d(
    kc_ini: float = 0.40,
    kc_mid: float = 1.15,
    kc_end: float = 0.50,
    L_ini: int = 15,
    L_dev: int = 15,
    L_mid: int = 40,
    L_late: int = 20,
) -> np.ndarray:
    """Standard FAO-56 Kc curve for early-cycle (90-day) soybean."""
    return fao56_kc_curve(kc_ini, kc_mid, kc_end, L_ini, L_dev, L_mid, L_late)


def fao56_kc_5stage_step(
    kc_ini: float,
    kc_dev: float,
    kc_mid: float,
    kc_late: float,
    kc_harvest: float,
    L_ini: int,
    L_dev: int,
    L_mid: int,
    L_late: int,
    L_harvest: int,
) -> np.ndarray:
    """5-stage step-function Kc curve (Embrapa convention, 90-day soybean).

    Each phenological phase has a constant Kc value (no linear ramp). This is
    the simplified time-averaged form discussed in FAO-56 Chapter 6 (Allen
    et al. 1998) and adopted by Brazilian soybean agronomy (Embrapa Soja),
    in which the development and late stages are represented by single
    representative Kc values rather than ramps. See also Steduto et al.
    (2012), FAO-66, for the soybean response function used here.

    Default values for early-cycle (90-day) soybean in MATOPIBA:
        L_ini=15  Kc=0.40   (initial / emergence)
        L_dev=15  Kc=0.80   (development / canopy expansion)
        L_mid=40  Kc=1.15   (mid-season / flowering-grain fill)
        L_late=15 Kc=0.80   (late-season / maturation)
        L_harvest=5 Kc=0.50 (harvest / senescence)

    Returns
    -------
    np.ndarray
        Vector of length ``L_ini + L_dev + L_mid + L_late + L_harvest``.
    """
    if min(L_ini, L_dev, L_mid, L_late, L_harvest) < 0:
        raise ValueError("Stage durations must be non-negative.")
    n = L_ini + L_dev + L_mid + L_late + L_harvest
    if n == 0:
        return np.array([], dtype=float)

    kc = np.empty(n, dtype=float)
    s = 0
    kc[s : s + L_ini] = kc_ini
    s += L_ini
    kc[s : s + L_dev] = kc_dev
    s += L_dev
    kc[s : s + L_mid] = kc_mid
    s += L_mid
    kc[s : s + L_late] = kc_late
    s += L_late
    kc[s : s + L_harvest] = kc_harvest
    return kc


def soybean_kc_90d_step(
    kc_ini: float = 0.40,
    kc_dev: float = 0.80,
    kc_mid: float = 1.15,
    kc_late: float = 0.80,
    kc_harvest: float = 0.50,
    L_ini: int = 15,
    L_dev: int = 15,
    L_mid: int = 40,
    L_late: int = 15,
    L_harvest: int = 5,
) -> np.ndarray:
    """5-stage step Kc for early-cycle (90-day) soybean per advisor spec."""
    return fao56_kc_5stage_step(
        kc_ini=kc_ini, kc_dev=kc_dev, kc_mid=kc_mid,
        kc_late=kc_late, kc_harvest=kc_harvest,
        L_ini=L_ini, L_dev=L_dev, L_mid=L_mid,
        L_late=L_late, L_harvest=L_harvest,
    )


def kc_daily_from_crop_dict(
    crop: dict,
    L_ini: int,
    L_dev: int,
    L_mid: int,
    L_late: int,
    range_mode: str = "mean",
) -> np.ndarray:
    """Build daily Kc curve from a FAO-56 crop dictionary.

    Parameters
    ----------
    crop : dict
        FAO-56 crop entry with `kc.ini`, `kc.mid`, `kc.end`. Each value can
        be a scalar or a {"low", "high"} range.
    L_ini, L_dev, L_mid, L_late : int
        Stage durations in days.
    range_mode : {"mean", "low", "high"}
        How to resolve range values (used when FAO Table 12 specifies a range).
    """
    def _resolve(v):
        if isinstance(v, dict) and "low" in v and "high" in v:
            if range_mode == "mean":
                return (v["low"] + v["high"]) / 2.0
            if range_mode == "low":
                return v["low"]
            if range_mode == "high":
                return v["high"]
            raise ValueError(f"range_mode inválido: {range_mode}")
        return float(v)

    kc = crop["kc"] if "kc" in crop else crop
    return fao56_kc_curve(
        kc_ini=_resolve(kc["ini"]),
        kc_mid=_resolve(kc["mid"]),
        kc_end=_resolve(kc["end"]),
        L_ini=L_ini, L_dev=L_dev, L_mid=L_mid, L_late=L_late,
    )
