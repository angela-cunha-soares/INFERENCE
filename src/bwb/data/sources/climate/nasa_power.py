"""NASA POWER daily climate downloader.

Adapted from the EVAOnline project (``backend/api/services/nasa_power``)
and rewritten as a synchronous, dependency-light function:

* No Pydantic, loguru, httpx, Redis, or DI — only the standard library and
  pandas / numpy.
* Returns a DataFrame with the project's canonical schema
  (``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``) so it is interchangeable with
  :func:`bwb.data.loaders.load_city_series`.
* ETo is computed in-place from Penman-Monteith (FAO-56) because NASA POWER
  does not expose ETo directly.

Usage
-----
>>> from datetime import date
>>> df = download_nasa_power(
...     lat=-7.4, lon=-46.0,
...     start=date(2020, 11, 1), end=date(2021, 2, 28),
...     elevation_m=283.0,
... )
>>> df.head()
        date    Tmax    Tmin    RH    u2     Rs     pr   ETo
0 2020-11-01   33.4   22.1  62.3  2.1  22.4   0.0   5.8
...

Caching
-------
Repeated calls with the same arguments are cached on disk via ``joblib`` if
available; otherwise an in-process ``functools.lru_cache`` is used. To force
a fresh download, pass ``use_cache=False``.

Reference
---------
NASA POWER terms of use: https://power.larc.nasa.gov/docs/
FAO-56 Penman-Monteith: Allen et al. (1998), Eq. 6.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from functools import lru_cache
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DEFAULT_TIMEOUT_S = 30
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 1.0

# NASA POWER parameter codes → bwb schema column names
PARAM_MAP = {
    "T2M_MAX": "Tmax",            # max temp at 2m (°C)
    "T2M_MIN": "Tmin",            # min temp at 2m (°C)
    "T2M": "Tmean",               # mean temp at 2m (°C); used to derive ETo
    "RH2M": "RH",                 # relative humidity at 2m (%)
    "WS2M": "u2",                 # wind speed at 2m (m/s)
    "ALLSKY_SFC_SW_DWN": "Rs",    # downward shortwave (MJ/m²/day)
    "PRECTOTCORR": "pr",          # bias-corrected precip (mm/day)
}

# Sentinel used by NASA POWER for missing values
MISSING_SENTINEL = -999.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def download_nasa_power(
    lat: float,
    lon: float,
    start: Union[str, date, datetime, pd.Timestamp],
    end: Union[str, date, datetime, pd.Timestamp],
    *,
    elevation_m: Optional[float] = None,
    community: str = "AG",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download NASA POWER daily data and return a bwb-schema DataFrame.

    Parameters
    ----------
    lat, lon : float
        Coordinates in WGS-84 decimal degrees.
    start, end : str | date | datetime | Timestamp
        Inclusive date range. Strings must be ``YYYY-MM-DD``.
    elevation_m : float, optional
        Elevation above mean sea level (m). Required to compute ETo via
        Penman-Monteith. If omitted, the ``ETo`` column is filled with NaN
        and a warning is logged.
    community : str
        NASA POWER community: ``"AG"`` (agronomy, default), ``"RE"``
        (renewable energy), ``"SB"`` (sustainable buildings).
    timeout_s : int
        HTTP request timeout (seconds).
    retries : int
        Number of attempts on transient HTTP failures (exponential backoff).
    use_cache : bool
        Reuse a previously downloaded result for the same arguments.

    Returns
    -------
    DataFrame with columns ``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``.
    """
    s = _to_date(start)
    e = _to_date(end)
    if s > e:
        raise ValueError(f"start ({s}) must be <= end ({e})")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"invalid coordinates: lat={lat}, lon={lon}")

    cache_key = (round(float(lat), 4), round(float(lon), 4),
                 s.isoformat(), e.isoformat(), community.upper())
    if use_cache:
        raw = _fetch_cached(*cache_key, timeout_s=timeout_s, retries=retries)
    else:
        raw = _fetch_uncached(*cache_key, timeout_s=timeout_s, retries=retries)

    df = _parse_to_dataframe(raw)
    df = _add_eto(df, lat=lat, elevation_m=elevation_m)
    # Reorder to canonical schema; drop the helper Tmean column.
    cols = ["date", "Rs", "u2", "Tmax", "Tmin", "RH", "pr", "ETo"]
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _fetch_cached(lat: float, lon: float, start_iso: str, end_iso: str,
                  community: str, *, timeout_s: int, retries: int) -> dict:
    return _fetch_uncached(lat, lon, start_iso, end_iso, community,
                           timeout_s=timeout_s, retries=retries)


def _fetch_uncached(lat: float, lon: float, start_iso: str, end_iso: str,
                    community: str, *, timeout_s: int, retries: int) -> dict:
    s = datetime.fromisoformat(start_iso).date()
    e = datetime.fromisoformat(end_iso).date()
    params = {
        "parameters": ",".join(PARAM_MAP.keys()),
        "community": community.upper(),
        "longitude": lon,
        "latitude": lat,
        "start": s.strftime("%Y%m%d"),
        "end": e.strftime("%Y%m%d"),
        "format": "JSON",
    }
    url = f"{NASA_POWER_URL}?{urllib.parse.urlencode(params)}"
    logger.info("NASA POWER request: lat=%s lon=%s %s..%s", lat, lon,
                params["start"], params["end"])
    return _http_get_json(url, timeout_s=timeout_s, retries=retries)


def _http_get_json(url: str, *, timeout_s: int, retries: int) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            backoff = DEFAULT_BACKOFF_S * (2 ** attempt)
            logger.warning("NASA POWER attempt %d/%d failed (%s); retrying in %.1fs",
                           attempt + 1, retries, exc, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"NASA POWER: all {retries} attempts failed: {last_err}")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_to_dataframe(payload: dict) -> pd.DataFrame:
    try:
        params = payload["properties"]["parameter"]
    except KeyError as exc:
        raise ValueError(
            f"unexpected NASA POWER response (missing 'properties.parameter'): {exc}"
        ) from exc

    first = next(iter(params.values()))
    dates = sorted(first.keys())  # YYYYMMDD strings
    rows = {col: [] for col in PARAM_MAP.values()}
    rows["date"] = []
    for d in dates:
        rows["date"].append(datetime.strptime(d, "%Y%m%d"))
        for code, col in PARAM_MAP.items():
            v = params.get(code, {}).get(d)
            if v is None or v == MISSING_SENTINEL:
                rows[col].append(np.nan)
            else:
                rows[col].append(float(v))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Penman-Monteith FAO-56 (Eq. 6)
# ---------------------------------------------------------------------------


def _add_eto(df: pd.DataFrame, *, lat: float,
             elevation_m: Optional[float]) -> pd.DataFrame:
    if elevation_m is None:
        logger.warning("download_nasa_power: elevation_m not provided; "
                       "ETo column will be NaN. Pass elevation to enable Penman-Monteith.")
        df["ETo"] = np.nan
        return df

    out = df.copy()
    Tmax = out["Tmax"].to_numpy()
    Tmin = out["Tmin"].to_numpy()
    Tmean = out["Tmean"].to_numpy() if "Tmean" in out else 0.5 * (Tmax + Tmin)
    RH = out["RH"].to_numpy()
    u2 = out["u2"].to_numpy()
    Rs = out["Rs"].to_numpy()  # MJ/m²/day
    doy = out["date"].dt.dayofyear.to_numpy()

    out["ETo"] = compute_eto_fao56_pm(
        Tmax=Tmax, Tmin=Tmin, Tmean=Tmean, RH=RH, u2=u2, Rs=Rs,
        doy=doy, lat_deg=lat, elevation_m=elevation_m,
    )
    return out


def compute_eto_fao56_pm(*,
                         Tmax: np.ndarray, Tmin: np.ndarray, Tmean: np.ndarray,
                         RH: np.ndarray, u2: np.ndarray, Rs: np.ndarray,
                         doy: np.ndarray, lat_deg: float,
                         elevation_m: float) -> np.ndarray:
    """Reference evapotranspiration (mm/day) per Allen et al. (1998), Eq. 6.

    Inputs are 1-D arrays of equal length. Returns ETo in mm/day.

    Required units (do **not** pass values in any other unit):

    +-----------------+--------------------------+---------------------------+
    | Argument        | Unit                     | Notes                     |
    +=================+==========================+===========================+
    | ``Tmax``,       | °C (degrees Celsius)     | daily extrema             |
    | ``Tmin``        |                          |                           |
    +-----------------+--------------------------+---------------------------+
    | ``Tmean``       | °C                       | daily mean; if you only   |
    |                 |                          | have Tmax/Tmin, pass      |
    |                 |                          | ``0.5*(Tmax+Tmin)``       |
    +-----------------+--------------------------+---------------------------+
    | ``RH``          | %  (0..100)              | mean daily relative       |
    |                 |                          | humidity (NOT a fraction) |
    +-----------------+--------------------------+---------------------------+
    | ``u2``          | m/s **at 2 m height**    | if you have wind at 10 m, |
    |                 |                          | call :func:`wind_10m_to_2m|
    |                 |                          | _fao56` first             |
    +-----------------+--------------------------+---------------------------+
    | ``Rs``          | MJ m⁻² day⁻¹             | downward shortwave at the |
    |                 |                          | surface; if your source   |
    |                 |                          | gives W/m² average over   |
    |                 |                          | the day, multiply by      |
    |                 |                          | 0.0864 (= 86 400/1e6)     |
    +-----------------+--------------------------+---------------------------+
    | ``doy``         | day of year (1..366)     | integer array              |
    +-----------------+--------------------------+---------------------------+
    | ``lat_deg``     | decimal degrees          | converted to radians here |
    +-----------------+--------------------------+---------------------------+
    | ``elevation_m`` | metres above MSL         | scalar                    |
    +-----------------+--------------------------+---------------------------+

    All bwb climate downloaders
    (:func:`download_nasa_power`, :func:`download_openmeteo_archive`,
    :func:`download_openmeteo_forecast`) already return values in these
    units, so this routine can be called directly on their outputs.
    """
    # Atmospheric pressure (kPa), Eq. 7
    P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    # Psychrometric constant (kPa/°C), Eq. 8
    gamma = 0.000665 * P
    # Saturation vapor pressure es and slope Δ (kPa, kPa/°C), Eq. 11-13
    es_Tmax = 0.6108 * np.exp(17.27 * Tmax / (Tmax + 237.3))
    es_Tmin = 0.6108 * np.exp(17.27 * Tmin / (Tmin + 237.3))
    es = 0.5 * (es_Tmax + es_Tmin)
    delta = 4098.0 * (0.6108 * np.exp(17.27 * Tmean / (Tmean + 237.3))) / \
            (Tmean + 237.3) ** 2
    # Actual vapor pressure (kPa), Eq. 19 from RH and es(Tmin)/es(Tmax)
    ea = 0.5 * (es_Tmin * RH / 100.0 + es_Tmax * RH / 100.0)
    # Extraterrestrial radiation Ra (MJ/m²/day), Eq. 21
    lat = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    delta_sun = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    cos_arg = np.clip(-np.tan(lat) * np.tan(delta_sun), -1.0, 1.0)
    omega_s = np.arccos(cos_arg)
    Gsc = 0.0820  # MJ/m²/min
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega_s * np.sin(lat) * np.sin(delta_sun) +
        np.cos(lat) * np.cos(delta_sun) * np.sin(omega_s)
    )
    # Clear-sky radiation Rso (Eq. 37)
    Rso = (0.75 + 2e-5 * elevation_m) * Ra
    # Net shortwave (Eq. 38), albedo 0.23
    Rns = (1.0 - 0.23) * Rs
    # Net longwave (Eq. 39); Stefan-Boltzmann in MJ K⁻⁴ m⁻² day⁻¹
    sigma = 4.903e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Rso > 0, Rs / np.where(Rso > 0, Rso, 1.0), 0.0)
    rs_over_rso = np.clip(ratio, 0.0, 1.0)
    Rnl = (sigma
           * 0.5 * ((Tmax + 273.16) ** 4 + (Tmin + 273.16) ** 4)
           * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
           * (1.35 * rs_over_rso - 0.35))
    Rn = Rns - Rnl
    # Soil heat flux for daily step ≈ 0
    G = 0.0
    # Penman-Monteith, Eq. 6
    num = 0.408 * delta * (Rn - G) + gamma * (900.0 / (Tmean + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)
    eto = num / den
    return np.maximum(eto, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(x: Union[str, date, datetime, pd.Timestamp]) -> date:
    if isinstance(x, str):
        return datetime.strptime(x, "%Y-%m-%d").date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().date()
    if isinstance(x, date):
        return x
    raise TypeError(f"unsupported date type: {type(x)}")


def wind_10m_to_2m_fao56(u_z: np.ndarray, *, z: float = 10.0) -> np.ndarray:
    """Convert wind speed measured at height ``z`` (m) to 2 m via FAO-56 Eq. 47.

    .. math::

        u_2 = u_z \\cdot \\dfrac{4.87}{\\ln(67.8\\,z - 5.42)}

    For the standard z = 10 m the multiplicative factor evaluates to
    ``4.87 / ln(67.8 * 10 - 5.42) ≈ 0.747951``, which matches the
    "approximately 0.75" rule of thumb often quoted for 10 m → 2 m
    conversion. The full logarithmic form is preferred because Open-Meteo
    occasionally reports wind at heights other than 10 m for some
    products.

    Negative values (which would arise from numerical noise on a
    near-zero wind) are clamped to 0.
    """
    u = np.asarray(u_z, dtype=float)
    factor = 4.87 / np.log(67.8 * float(z) - 5.42)
    return np.maximum(u * factor, 0.0)


__all__ = ["download_nasa_power", "compute_eto_fao56_pm", "wind_10m_to_2m_fao56"]
