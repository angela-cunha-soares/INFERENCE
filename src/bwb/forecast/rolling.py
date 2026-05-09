"""Operational rolling 5-day irrigation forecast.

Given (location, crop, planting_date, forecast_date), produce a probabilistic
5-day-ahead forecast of:
  * daily irrigation requirement (mm)
  * soil water content trajectory (mm)
  * probability that irrigation is needed each day
  * 90% prediction intervals on cumulative irrigation over the horizon

The algorithm is a thin layer over the existing climatological resampling
machinery (:mod:`bwb.forecast.climatological`):

1. **State reconstruction.** From ``planting_date`` to ``forecast_date``,
   run the deterministic FAO-56 daily water balance (:mod:`bwb.models.fao56`)
   driven by observed P and ETo. The end-state ``SW(forecast_date)`` is the
   initial condition for the forecast.

2. **Horizon Monte-Carlo.** For days ``forecast_date+1 .. forecast_date+H``,
   draw ``n_simulations`` weight vectors ``w ~ Dir(alpha)`` from the
   user-supplied Dirichlet prior (preferably the climatologically informed
   one persisted by :func:`bwb.forecast.climatological.run_sequential_forecast`).
   For each draw, sample a class ``kappa ~ Cat(w)``, resample one historical
   season of that class, slice the H daily forcings starting at the
   appropriate day-of-cycle, and continue the FAO-56 balance from the
   reconstructed ``SW``.

3. **Aggregation.** Per-day quantiles of irrigation, probability of
   non-zero irrigation event, 90% interval on cumulative I over horizon.

This module is *crop-agnostic* and *location-agnostic* — the forecast date,
crop block, and observed climate are passed in as arguments. For Brazilian
cities we drive it from Xavier reanalysis; for other locations a daily
P/ETo time series is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from bwb.models.fao56 import daily_water_balance
from bwb.forecast.climatological import (
    N_CLASSES, classify_seasons, extract_historical_seasons,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RollingForecast:
    """Result of a single rolling 5-day forecast run."""

    forecast_date: pd.Timestamp
    planting_date: pd.Timestamp
    horizon_days: int
    day_of_cycle: int                # day index within the crop cycle (0-based)

    # State at forecast_date (deterministic reconstruction from observations)
    SW_today_mm: float
    I_to_date_mm: float              # cumulative I from planting_date..forecast_date
    awc_mm: float

    # Forecast distribution over horizon (n_sim, horizon)
    SW_horizon: np.ndarray           # mm, shape (n_sim, horizon)
    I_horizon: np.ndarray            # mm/day
    ETc_horizon: np.ndarray          # mm/day
    sampled_classes: np.ndarray      # shape (n_sim,)
    sampled_years: np.ndarray        # shape (n_sim,)

    # Convenience aggregates
    I_total_horizon_q05: float = 0.0
    I_total_horizon_q50: float = 0.0
    I_total_horizon_q95: float = 0.0
    p_irrigate_tomorrow: float = 0.0
    I_tomorrow_q05: float = 0.0
    I_tomorrow_q50: float = 0.0
    I_tomorrow_q95: float = 0.0

    # Optional verification (filled when forecast_date is in the past and
    # observed data is available for the horizon)
    obs_I_horizon: Optional[np.ndarray] = None     # shape (horizon,)
    obs_SW_horizon: Optional[np.ndarray] = None
    obs_I_total_horizon: Optional[float] = None
    crps_I_total: Optional[float] = None

    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------

def reconstruct_state_at_date(
    history_df: pd.DataFrame,
    planting_date: pd.Timestamp,
    forecast_date: pd.Timestamp,
    kc_curve: np.ndarray,
    awc_mm: float,
    mad: float,
    p_eff_factor: float = 0.8,
    sw_init: Optional[float] = None,
) -> tuple[float, float, np.ndarray]:
    """Run the FAO-56 deterministic balance from planting to forecast_date.

    Returns
    -------
    sw_today : float
        Soil water content (mm) at the end of forecast_date.
    cumulative_irrigation : float
        Total I (mm) from planting through forecast_date.
    sw_history : np.ndarray
        Daily SW trajectory from planting to forecast_date (length n).
    """
    mask = (history_df["date"] >= planting_date) & (history_df["date"] <= forecast_date)
    window = history_df.loc[mask].sort_values("date").reset_index(drop=True)
    n = len(window)
    if n == 0:
        # No history available -> start at field capacity
        return float(sw_init) if sw_init is not None else float(awc_mm), 0.0, np.array([])

    # Ensure kc_curve is long enough
    if len(kc_curve) < n:
        # pad with last value (e.g. harvest Kc)
        kc = np.concatenate([kc_curve, np.full(n - len(kc_curve), kc_curve[-1])])
    else:
        kc = kc_curve[:n]

    eto = window["ETo"].to_numpy(dtype=float)
    pr = window["pr"].to_numpy(dtype=float)
    wb = daily_water_balance(
        eto=eto, pr=pr, kc=kc,
        awc=awc_mm, mad=mad, p_eff_factor=p_eff_factor,
        sw_init=sw_init,
    )
    return float(wb.SW[-1]), float(wb.I.sum()), wb.SW


# ---------------------------------------------------------------------------
# Horizon Monte-Carlo
# ---------------------------------------------------------------------------

def _resample_horizon(
    historical_seasons: dict,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Draw a (class, year) pair given Dirichlet-sampled weights."""
    klass = int(rng.choice(N_CLASSES, p=weights))
    candidates = [year for year, season in historical_seasons.items()
                  if season["klass"] == klass]
    if not candidates:
        # fall back to any year
        candidates = list(historical_seasons.keys())
    year = int(rng.choice(candidates))
    return klass, year


def horizon_monte_carlo(
    historical_seasons: dict,
    sw_today: float,
    day_of_cycle: int,
    horizon_days: int,
    kc_curve: np.ndarray,
    awc_mm: float,
    mad: float,
    alpha: np.ndarray,
    n_simulations: int,
    p_eff_factor: float = 0.8,
    random_seed: Optional[int] = None,
    horizon_pr_ensemble: Optional[np.ndarray] = None,
    horizon_eto_ensemble: Optional[np.ndarray] = None,
) -> dict:
    """Run the horizon-step Monte-Carlo on top of a reconstructed SW state.

    Two operational modes:

    * **Climatology resampling** (default; ``horizon_pr_ensemble`` is None):
      Each simulation draws a Dirichlet weight, samples a class, then resamples
      a complete historical season of that class as the horizon weather. This
      is the *only* mode usable for backtests against reanalysis (no NWP
      archive available pre-2000).

    * **NWP override / hybrid** (``horizon_pr_ensemble`` provided): Each
      simulation draws one ensemble member from the supplied NWP forecast.
      Useful for *operational* deployment where ECMWF/GFS/OpenMeteo provide
      tightly-calibrated 5-day forecasts. The Dirichlet posterior is still
      tracked (for class diagnostics) but does not drive the weather.

      Accepted shapes:
        - (n_members, horizon_days): full ensemble — sampled with replacement
        - (horizon_days,): single deterministic forecast — replicated to all
          n_simulations (Monte-Carlo then varies only over the climatology
          spread of historical analogues for analogous-class diagnostics)

    Returns dict with arrays SW_horizon, I_horizon, ETc_horizon,
    sampled_classes, sampled_years.
    """
    rng = np.random.default_rng(random_seed)

    weights_samples = rng.dirichlet(alpha, size=n_simulations)
    SW = np.empty((n_simulations, horizon_days), dtype=float)
    I = np.empty_like(SW)
    ETc = np.empty_like(SW)
    sampled_classes = np.empty(n_simulations, dtype=int)
    sampled_years = np.empty(n_simulations, dtype=int)

    kc_horizon_start = day_of_cycle + 1
    if kc_horizon_start + horizon_days > len(kc_curve):
        kc = np.concatenate([
            kc_curve[kc_horizon_start:],
            np.full(kc_horizon_start + horizon_days - len(kc_curve),
                    kc_curve[-1]),
        ])
    else:
        kc = kc_curve[kc_horizon_start : kc_horizon_start + horizon_days]

    # Pre-process the NWP ensemble if provided
    use_nwp = horizon_pr_ensemble is not None
    if use_nwp:
        if horizon_eto_ensemble is None:
            raise ValueError(
                "horizon_eto_ensemble must be provided when horizon_pr_ensemble is")
        pr_ens = np.asarray(horizon_pr_ensemble, dtype=float)
        eto_ens = np.asarray(horizon_eto_ensemble, dtype=float)
        if pr_ens.ndim == 1:
            pr_ens = pr_ens[np.newaxis, :]
            eto_ens = eto_ens[np.newaxis, :]
        if pr_ens.shape[1] != horizon_days or eto_ens.shape[1] != horizon_days:
            raise ValueError(
                f"NWP ensemble second axis must equal horizon_days={horizon_days}; "
                f"got pr {pr_ens.shape}, eto {eto_ens.shape}")

    for i in range(n_simulations):
        klass, year = _resample_horizon(historical_seasons, weights_samples[i], rng)

        if use_nwp:
            m = int(rng.integers(0, pr_ens.shape[0]))
            P_horizon = pr_ens[m]
            ETo_horizon = eto_ens[m]
        else:
            season = historical_seasons[year]
            slice_start = kc_horizon_start
            if slice_start + horizon_days <= len(season["P_daily"]):
                P_horizon = season["P_daily"][slice_start : slice_start + horizon_days]
                ETo_horizon = season["ETo_daily"][slice_start : slice_start + horizon_days]
            else:
                need = slice_start + horizon_days - len(season["P_daily"])
                P_horizon = np.concatenate([
                    season["P_daily"][slice_start:],
                    np.full(need, season["P_daily"][-1]),
                ])
                ETo_horizon = np.concatenate([
                    season["ETo_daily"][slice_start:],
                    np.full(need, season["ETo_daily"][-1]),
                ])

        wb = daily_water_balance(
            eto=ETo_horizon, pr=P_horizon, kc=kc,
            awc=awc_mm, mad=mad, p_eff_factor=p_eff_factor,
            sw_init=sw_today,
        )
        SW[i] = wb.SW
        I[i] = wb.I
        ETc[i] = wb.ETc
        sampled_classes[i] = klass
        sampled_years[i] = year

    return {
        "SW": SW, "I": I, "ETc": ETc,
        "sampled_classes": sampled_classes,
        "sampled_years": sampled_years,
    }


# ---------------------------------------------------------------------------
# Top-level rolling 5-day forecast
# ---------------------------------------------------------------------------

def rolling_5day_forecast(
    history_df: Optional[pd.DataFrame] = None,
    planting_date: Optional[pd.Timestamp] = None,
    forecast_date: Optional[pd.Timestamp] = None,
    kc_curve: Optional[np.ndarray] = None,
    awc_mm: Optional[float] = None,
    *,
    cycle_days: int = 90,
    horizon_days: int = 5,
    mad: float = 0.55,
    p_eff_factor: float = 0.8,
    alpha: Optional[np.ndarray] = None,
    n_simulations: int = 500,
    method: str = "spei",
    random_seed: Optional[int] = 42,
    sw_init: Optional[float] = None,
    horizon_pr_ensemble: Optional[np.ndarray] = None,
    horizon_eto_ensemble: Optional[np.ndarray] = None,
    weather_sources: Optional[Mapping[str, pd.DataFrame]] = None,
    fusion_lat: Optional[float] = None,
    fusion_elevation_m: Optional[float] = None,
) -> RollingForecast:
    """Produce a rolling H-day operational irrigation forecast.

    Parameters
    ----------
    history_df : DataFrame with columns ['date', 'pr', 'ETo']
        Daily climate observations covering at least 1961..forecast_date.
        Used both for reconstructing the current SW state (post-planting
        observed window) and as the source of historical-season analogues
        for the horizon Monte-Carlo (pre-forecast window).
    planting_date, forecast_date : Timestamp
        Crop cycle anchors.
    kc_curve : np.ndarray of length cycle_days
        Per-day crop coefficient curve (FAO-56).
    awc_mm : float
        Available water capacity in mm for the root zone.
    cycle_days : int
        Total crop cycle length (default 90 for early-soybean).
    horizon_days : int
        Forecast horizon (default 5 days).
    mad : float
        Management allowed depletion fraction (default 0.55, FAO-56 soybean).
    alpha : array (3,), optional
        Dirichlet prior on dry/normal/wet weights. Defaults to climatological
        counts from history_df pre-forecast_date plus a Laplace smoother.
    n_simulations : int
        Monte-Carlo ensemble size for the horizon (default 500).
    sw_init : float, optional
        Initial SW at planting (default = awc_mm, full at planting).
    horizon_pr_ensemble, horizon_eto_ensemble : array, optional
        Operational NWP override. If provided, the horizon weather is sampled
        from these arrays instead of resampled from climatological analogues.
        Each is either ``(n_members, horizon_days)`` (ECMWF/GFS ensemble) or
        ``(horizon_days,)`` (single deterministic forecast). When omitted
        (default), the model falls back to climatology resampling — required
        for backtest validation against pre-NWP reanalysis archives.
    weather_sources : mapping of source name → DataFrame, optional
        Live multi-source mode for 2026 operational deployment. When provided,
        ``history_df`` is built by calling
        :func:`bwb.data.sources.climate.fuse_climate_sources` over the supplied
        sources (NASA POWER, Open-Meteo Archive, Open-Meteo Forecast). Each
        DataFrame must follow the canonical bwb schema. Mutually exclusive
        with passing a pre-built ``history_df`` — exactly one of the two
        must be provided. Requires ``fusion_lat`` and ``fusion_elevation_m``
        so ETo can be recomputed from the fused inputs.
    fusion_lat, fusion_elevation_m : float, optional
        Latitude (degrees) and elevation (m a.s.l.) for the fusion site.
        Required when ``weather_sources`` is provided.

    Returns
    -------
    RollingForecast
    """
    if (history_df is None) == (weather_sources is None):
        raise ValueError(
            "rolling_5day_forecast: pass exactly one of `history_df` or "
            "`weather_sources` (got neither or both)."
        )
    if weather_sources is not None:
        if fusion_lat is None or fusion_elevation_m is None:
            raise ValueError(
                "rolling_5day_forecast: `fusion_lat` and `fusion_elevation_m` "
                "are required when `weather_sources` is supplied."
            )
        from bwb.data.sources.climate import fuse_climate_sources
        history_df = fuse_climate_sources(
            weather_sources,
            lat=float(fusion_lat),
            elevation_m=float(fusion_elevation_m),
        )

    if planting_date is None or forecast_date is None or kc_curve is None or awc_mm is None:
        raise ValueError(
            "rolling_5day_forecast: planting_date, forecast_date, kc_curve "
            "and awc_mm are required."
        )

    history_df = history_df.copy()
    history_df["date"] = pd.to_datetime(history_df["date"])
    planting_date = pd.Timestamp(planting_date)
    forecast_date = pd.Timestamp(forecast_date)

    if forecast_date < planting_date:
        raise ValueError("forecast_date must be on or after planting_date")
    day_of_cycle = (forecast_date - planting_date).days
    if day_of_cycle >= cycle_days:
        raise ValueError(
            f"forecast_date is past harvest (day_of_cycle={day_of_cycle} "
            f">= cycle_days={cycle_days})"
        )

    # 1) Reconstruct SW(forecast_date) from observed P, ETo
    sw_today, i_to_date, sw_history = reconstruct_state_at_date(
        history_df, planting_date, forecast_date, kc_curve,
        awc_mm=awc_mm, mad=mad, p_eff_factor=p_eff_factor, sw_init=sw_init,
    )

    # 2) Build the climatology base for resampling: 1961..(forecast_year - 1)
    forecast_year = int(planting_date.year)
    # Use cycles starting at the same planting month/day, ending strictly
    # before the current cycle's planting year.
    stats_df, daily = extract_historical_seasons(
        history_df,
        planting_month=int(planting_date.month),
        planting_day=int(planting_date.day),
        cycle_days=cycle_days,
        until_year_exclusive=forecast_year,
    )
    classes, _meta = classify_seasons(stats_df, method=method)
    historical_seasons: dict[int, dict] = {
        int(year): {
            "klass": int(classes[int(year)]),
            "P_daily": daily[year]["pr"],
            "ETo_daily": daily[year]["ETo"],
        }
        for year in classes
    }

    # 3) Default alpha = climatological counts + Laplace smoother
    if alpha is None:
        counts = np.array([
            sum(1 for k in classes.values() if k == cls)
            for cls in range(N_CLASSES)
        ], dtype=float)
        alpha = counts + 1.0

    # 4) Horizon Monte-Carlo
    mc = horizon_monte_carlo(
        historical_seasons=historical_seasons,
        sw_today=sw_today,
        day_of_cycle=day_of_cycle,
        horizon_days=horizon_days,
        kc_curve=kc_curve,
        awc_mm=awc_mm,
        mad=mad,
        alpha=alpha,
        n_simulations=n_simulations,
        p_eff_factor=p_eff_factor,
        random_seed=random_seed,
        horizon_pr_ensemble=horizon_pr_ensemble,
        horizon_eto_ensemble=horizon_eto_ensemble,
    )

    # 5) Aggregates
    I_total = mc["I"].sum(axis=1)
    p_irr_tomorrow = float((mc["I"][:, 0] > 0).mean())
    I_tomorrow = mc["I"][:, 0]

    rf = RollingForecast(
        forecast_date=forecast_date,
        planting_date=planting_date,
        horizon_days=horizon_days,
        day_of_cycle=day_of_cycle,
        SW_today_mm=sw_today,
        I_to_date_mm=i_to_date,
        awc_mm=awc_mm,
        SW_horizon=mc["SW"],
        I_horizon=mc["I"],
        ETc_horizon=mc["ETc"],
        sampled_classes=mc["sampled_classes"],
        sampled_years=mc["sampled_years"],
        I_total_horizon_q05=float(np.percentile(I_total, 5)),
        I_total_horizon_q50=float(np.percentile(I_total, 50)),
        I_total_horizon_q95=float(np.percentile(I_total, 95)),
        p_irrigate_tomorrow=p_irr_tomorrow,
        I_tomorrow_q05=float(np.percentile(I_tomorrow, 5)),
        I_tomorrow_q50=float(np.percentile(I_tomorrow, 50)),
        I_tomorrow_q95=float(np.percentile(I_tomorrow, 95)),
    )

    # 6) If the horizon is in the past (backtest mode), fill verification
    horizon_end = forecast_date + pd.Timedelta(days=horizon_days)
    obs_window = history_df.loc[
        (history_df["date"] > forecast_date)
        & (history_df["date"] <= horizon_end)
    ].sort_values("date").reset_index(drop=True)
    if len(obs_window) == horizon_days:
        # Run deterministic FAO-56 over the observed horizon for "ground truth" I
        kc_h = kc_curve[day_of_cycle + 1 : day_of_cycle + 1 + horizon_days]
        wb_obs = daily_water_balance(
            eto=obs_window["ETo"].to_numpy(dtype=float),
            pr=obs_window["pr"].to_numpy(dtype=float),
            kc=kc_h,
            awc=awc_mm, mad=mad, p_eff_factor=p_eff_factor,
            sw_init=sw_today,
        )
        rf.obs_I_horizon = wb_obs.I
        rf.obs_SW_horizon = wb_obs.SW
        rf.obs_I_total_horizon = float(wb_obs.I.sum())
        rf.crps_I_total = _crps_ensemble_1d(I_total, rf.obs_I_total_horizon)

    rf.metadata = {
        "n_simulations": n_simulations,
        "alpha": alpha.tolist(),
        "method": method,
        "cycle_days": cycle_days,
        "n_historical_seasons": len(historical_seasons),
    }
    return rf


def _crps_ensemble_1d(samples: np.ndarray, observation: float) -> float:
    """Empirical CRPS for 1-D ensemble (Hersbach 2000 estimator)."""
    samples = np.sort(np.asarray(samples, dtype=float))
    n = len(samples)
    cdf = np.arange(1, n + 1) / n
    indicator = (samples >= observation).astype(float)
    return float(np.trapezoid((cdf - indicator) ** 2, samples))
