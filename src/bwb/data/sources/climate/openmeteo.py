"""Open-Meteo Archive + Forecast daily climate downloaders.

Adapted from the EVAOnline project
(``backend/api/services/openmeteo_archive`` and ``openmeteo_forecast``)
and rewritten as synchronous, dependency-light functions:

* No ``openmeteo_requests`` / flatbuffers / protobuf — we hit the public
  JSON endpoint directly with ``urllib.request``.
* No Pydantic, loguru, httpx, Redis, requests_cache, retry_requests, or DI.
* Returns DataFrames with the project's canonical schema
  (``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``) so they are interchangeable
  with :func:`bwb.data.loaders.load_city_series` and
  :func:`bwb.data.sources.climate.nasa_power.download_nasa_power`.
* ETo comes pre-computed from Open-Meteo's
  ``et0_fao_evapotranspiration`` field (no Penman-Monteith needed here).

Two endpoints are exposed:

* :func:`download_openmeteo_archive` — historical reanalysis
  (1940-01-01 to today − 2 days), updates daily.
* :func:`download_openmeteo_forecast` — past 29 days + next 5 days
  (rolling window). Used for the operational NWP-override mode of
  :func:`bwb.forecast.rolling.rolling_5day_forecast`.

Wind is requested in m/s and converted from 10 m to 2 m via FAO-56 Eq. 47.

Reference
---------
Open-Meteo terms of use: https://open-meteo.com/en/terms (CC-BY 4.0).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import pandas as pd

from bwb.data.sources.climate.nasa_power import wind_10m_to_2m_fao56

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_S = 30
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 1.0

# Open-Meteo daily variables → bwb canonical column names.
# Order matters because Open-Meteo returns one array per variable.
DAILY_VARS = [
    "temperature_2m_max",          # Tmax (°C)
    "temperature_2m_min",          # Tmin (°C)
    "temperature_2m_mean",         # Tmean (°C, helper)
    "relative_humidity_2m_mean",   # RH (%)
    "wind_speed_10m_mean",         # wind at 10 m (m/s); → u2 via FAO-56 Eq. 47
    "shortwave_radiation_sum",     # Rs (MJ/m²/day)
    "precipitation_sum",           # pr (mm/day)
    "et0_fao_evapotranspiration",  # ETo (mm/day, Open-Meteo's PM)
]

OM_TO_BWB = {
    "temperature_2m_max": "Tmax",
    "temperature_2m_min": "Tmin",
    "temperature_2m_mean": "Tmean",
    "relative_humidity_2m_mean": "RH",
    # wind_speed_10m_mean is converted then renamed to u2 below
    "shortwave_radiation_sum": "Rs",
    "precipitation_sum": "pr",
    "et0_fao_evapotranspiration": "ETo",
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def download_openmeteo_archive(
    lat: float,
    lon: float,
    start: Union[str, date, datetime, pd.Timestamp],
    end: Union[str, date, datetime, pd.Timestamp],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download Open-Meteo Archive daily data and return a bwb DataFrame.

    Parameters
    ----------
    lat, lon : float
        Coordinates in WGS-84 decimal degrees.
    start, end : str | date | datetime | Timestamp
        Inclusive date range. Strings must be ``YYYY-MM-DD``.
        ``end`` must be at most ``today − 2 days`` (Archive constraint).
    timeout_s, retries : int
        HTTP behaviour.
    use_cache : bool
        Reuse a previously downloaded result for the same arguments.

    Returns
    -------
    DataFrame with columns ``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``.
    """
    s = _to_date(start)
    e = _to_date(end)
    _validate(lat, lon, s, e)
    max_archive = date.today() - timedelta(days=2)
    if e > max_archive:
        raise ValueError(
            f"Archive: end ({e}) must be <= today-2 ({max_archive}). "
            "Use download_openmeteo_forecast for more recent data."
        )

    key = (round(float(lat), 4), round(float(lon), 4),
           s.isoformat(), e.isoformat())
    payload = _archive_cached(*key, timeout_s=timeout_s, retries=retries) \
        if use_cache else _archive_uncached(*key, timeout_s=timeout_s,
                                            retries=retries)
    return _parse_to_dataframe(payload)


def download_openmeteo_forecast(
    lat: float,
    lon: float,
    *,
    past_days: int = 29,
    forecast_days: int = 5,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    use_cache: bool = False,  # forecast moves with time; default no cache
) -> pd.DataFrame:
    """Download Open-Meteo Forecast daily data (rolling window).

    Parameters
    ----------
    lat, lon : float
        Coordinates.
    past_days : int
        Days of recent past to include (0..29). Default 29 to bridge the
        2-day gap left by the Archive endpoint and provide overlap.
    forecast_days : int
        Days of future to include (0..16). Default 5 — matches the
        operational H=5 horizon used by ``rolling_5day_forecast``.
    timeout_s, retries : int
        HTTP behaviour.
    use_cache : bool
        Forecast data refreshes hourly, so cache is OFF by default.

    Returns
    -------
    DataFrame with columns ``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``.
    """
    if not (0 <= past_days <= 92):
        raise ValueError(f"past_days must be in [0, 92], got {past_days}")
    if not (0 <= forecast_days <= 16):
        raise ValueError(f"forecast_days must be in [0, 16], got {forecast_days}")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"invalid coordinates: lat={lat}, lon={lon}")

    key = (round(float(lat), 4), round(float(lon), 4),
           int(past_days), int(forecast_days))
    payload = _forecast_cached(*key, timeout_s=timeout_s, retries=retries) \
        if use_cache else _forecast_uncached(*key, timeout_s=timeout_s,
                                             retries=retries)
    return _parse_to_dataframe(payload)


# ---------------------------------------------------------------------------
# HTTP layer (Archive)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _archive_cached(lat: float, lon: float, start_iso: str, end_iso: str,
                    *, timeout_s: int, retries: int) -> dict:
    return _archive_uncached(lat, lon, start_iso, end_iso,
                             timeout_s=timeout_s, retries=retries)


def _archive_uncached(lat: float, lon: float, start_iso: str, end_iso: str,
                      *, timeout_s: int, retries: int) -> dict:
    params = _common_daily_params(lat, lon)
    params["start_date"] = start_iso
    params["end_date"] = end_iso
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params, doseq=True)}"
    logger.info("Open-Meteo Archive: lat=%s lon=%s %s..%s",
                lat, lon, start_iso, end_iso)
    return _http_get_json(url, timeout_s=timeout_s, retries=retries)


# ---------------------------------------------------------------------------
# HTTP layer (Forecast)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _forecast_cached(lat: float, lon: float, past_days: int,
                     forecast_days: int, *, timeout_s: int,
                     retries: int) -> dict:
    return _forecast_uncached(lat, lon, past_days, forecast_days,
                              timeout_s=timeout_s, retries=retries)


def _forecast_uncached(lat: float, lon: float, past_days: int,
                       forecast_days: int, *, timeout_s: int,
                       retries: int) -> dict:
    params = _common_daily_params(lat, lon)
    if past_days > 0:
        params["past_days"] = past_days
    # Open-Meteo always includes today; forecast_days is "additional days".
    params["forecast_days"] = max(forecast_days, 1)
    url = f"{FORECAST_URL}?{urllib.parse.urlencode(params, doseq=True)}"
    logger.info("Open-Meteo Forecast: lat=%s lon=%s past_days=%d forecast_days=%d",
                lat, lon, past_days, forecast_days)
    return _http_get_json(url, timeout_s=timeout_s, retries=retries)


def _common_daily_params(lat: float, lon: float) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }


def _http_get_json(url: str, *, timeout_s: int, retries: int) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            backoff = DEFAULT_BACKOFF_S * (2 ** attempt)
            logger.warning("Open-Meteo attempt %d/%d failed (%s); retrying in %.1fs",
                           attempt + 1, retries, exc, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"Open-Meteo: all {retries} attempts failed: {last_err}")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_to_dataframe(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        msg = payload.get("reason") or "missing 'daily' block"
        raise ValueError(f"unexpected Open-Meteo response: {msg}")

    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for om_col in DAILY_VARS:
        values = daily.get(om_col, [])
        if om_col == "wind_speed_10m_mean":
            # Open-Meteo always returns wind at 10 m; convert to 2 m
            # via FAO-56 Eq. 47 so the value matches the bwb canonical
            # ``u2`` column expected by Penman-Monteith.
            df["u2"] = wind_10m_to_2m_fao56(np.asarray(values, dtype=float))
        else:
            bwb_col = OM_TO_BWB[om_col]
            df[bwb_col] = pd.to_numeric(values, errors="coerce").astype(float) \
                if not isinstance(values, np.ndarray) else values.astype(float)

    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "Rs", "u2", "Tmax", "Tmin", "RH", "pr", "ETo"]]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(lat: float, lon: float, s: date, e: date) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"invalid coordinates: lat={lat}, lon={lon}")
    if s > e:
        raise ValueError(f"start ({s}) must be <= end ({e})")


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


__all__ = ["download_openmeteo_archive", "download_openmeteo_forecast"]
