"""Naive forecast baselines for skill-score comparison.

The climatological sequential forecast (:mod:`bwb.forecast.climatological`)
must demonstrate skill *above* simple alternatives -- otherwise its KGE
of 0.13 on the daily soil-water trajectory could be dismissed as ``just
climatology''. This module implements three reference baselines whose
CRPS is then used to compute the CRPS Skill Score (CRPSS):

    CRPSS = 1 - CRPS_model / CRPS_baseline

CRPSS > 0 means the model beats the baseline; CRPSS < 0 means the
baseline beats the model. CRPSS is the canonical proper-scoring-rule
skill score in operational meteorology and hydrology
(Wilks 2011; Hersbach 2000).

Baselines implemented
---------------------

* :func:`naive_climatology_forecast` -- the empirical distribution of
  seasonal totals from the last *N* historical cycles, ignoring any
  stratification or update.
* :func:`persistence_forecast` -- a degenerate distribution at the
  realised value of the most recent observed cycle.
* :func:`historical_resample_forecast` -- like the climatological mode
  but with a *uniform* (not Dirichlet) sampling probability across all
  past seasons; serves as the ``no-classifier'' baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from bwb.models.fao56 import daily_water_balance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crps_ensemble_1d(samples: np.ndarray, observation: float) -> float:
    """Hersbach-2000 CRPS estimator for a single observation."""
    s = np.asarray(samples, dtype=float)
    n = len(s)
    if n == 0:
        return float("nan")
    sorted_s = np.sort(s)
    weights = (2 * np.arange(1, n + 1) - n - 1) / (n ** 2)
    return float(np.mean(np.abs(s - observation)) - np.sum(weights * sorted_s))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class BaselineForecast:
    """Generic container for any baseline forecast distribution."""

    cycle: int
    method: str
    I_total_samples: np.ndarray   # (n_simulations,) -- seasonal irrigation
    metadata: dict


def naive_climatology_forecast(
    historical_seasons_pr: dict[int, np.ndarray],
    historical_seasons_eto: dict[int, np.ndarray],
    *,
    target_cycle: int,
    kc_curve: np.ndarray,
    awc: float,
    mad: float = 0.55,
    p_eff_factor: float = 0.8,
    n_simulations: int = 500,
    random_seed: Optional[int] = None,
) -> BaselineForecast:
    """Pool *all* historical cycles uniformly and resample.

    This is the simplest possible probabilistic forecast: it ignores the
    SPEI classification and the Dirichlet update, treating every past
    cycle as equally likely. If the full framework cannot beat this,
    the SPEI + Dirichlet machinery is not adding skill.

    Parameters
    ----------
    historical_seasons_pr, historical_seasons_eto : dict[year -> ndarray]
        Daily P and ETo for each past cycle (must have year < target_cycle).
    target_cycle : int
    kc_curve : ndarray of shape (cycle_days,)
    awc : float
        Available water capacity (mm).
    mad : float
        Management-allowed depletion fraction.
    p_eff_factor : float
        Effective precipitation factor.
    n_simulations : int
    random_seed : int, optional
    """
    rng = np.random.default_rng(random_seed)
    past_years = sorted(y for y in historical_seasons_pr.keys() if y < target_cycle)
    if not past_years:
        raise ValueError("No historical years before target_cycle")

    I_totals = np.empty(n_simulations, dtype=float)
    for i in range(n_simulations):
        y = int(rng.choice(past_years))
        wb = daily_water_balance(
            eto=historical_seasons_eto[y],
            pr=historical_seasons_pr[y],
            kc=kc_curve,
            awc=awc, mad=mad, p_eff_factor=p_eff_factor,
        )
        I_totals[i] = wb.I.sum()

    return BaselineForecast(
        cycle=target_cycle,
        method="naive_climatology",
        I_total_samples=I_totals,
        metadata={"n_past_years": len(past_years)},
    )


def persistence_forecast(
    historical_seasons_pr: dict[int, np.ndarray],
    historical_seasons_eto: dict[int, np.ndarray],
    *,
    target_cycle: int,
    kc_curve: np.ndarray,
    awc: float,
    mad: float = 0.55,
    p_eff_factor: float = 0.8,
    n_simulations: int = 500,
) -> BaselineForecast:
    """Predict that next cycle = last observed cycle.

    A degenerate forecast: the entire ``distribution'' collapses to the
    realised seasonal total of the most recent observed cycle. This is
    the simplest ``no-skill'' meteorological baseline.

    The output retains ``n_simulations`` samples (all identical) so the
    CRPS comparison is dimensionally consistent.
    """
    past_years = sorted(y for y in historical_seasons_pr.keys() if y < target_cycle)
    if not past_years:
        raise ValueError("No historical years before target_cycle")
    last_year = past_years[-1]

    wb = daily_water_balance(
        eto=historical_seasons_eto[last_year],
        pr=historical_seasons_pr[last_year],
        kc=kc_curve, awc=awc, mad=mad, p_eff_factor=p_eff_factor,
    )
    I_total_last = float(wb.I.sum())
    I_totals = np.full(n_simulations, I_total_last, dtype=float)

    return BaselineForecast(
        cycle=target_cycle,
        method="persistence",
        I_total_samples=I_totals,
        metadata={"reference_year": last_year},
    )


def long_term_mean_forecast(
    historical_seasons_pr: dict[int, np.ndarray],
    historical_seasons_eto: dict[int, np.ndarray],
    *,
    target_cycle: int,
    kc_curve: np.ndarray,
    awc: float,
    mad: float = 0.55,
    p_eff_factor: float = 0.8,
    n_simulations: int = 500,
) -> BaselineForecast:
    """Predict the long-term mean (single value) repeated n_simulations times.

    Even simpler than persistence: the forecast is the climatological
    mean of seasonal irrigation across all past cycles, with no
    uncertainty. Acts as the floor for any probabilistic CRPS
    comparison.
    """
    past_years = sorted(y for y in historical_seasons_pr.keys() if y < target_cycle)
    if not past_years:
        raise ValueError("No historical years before target_cycle")

    I_history = []
    for y in past_years:
        wb = daily_water_balance(
            eto=historical_seasons_eto[y],
            pr=historical_seasons_pr[y],
            kc=kc_curve, awc=awc, mad=mad, p_eff_factor=p_eff_factor,
        )
        I_history.append(wb.I.sum())
    mean_I = float(np.mean(I_history))
    I_totals = np.full(n_simulations, mean_I, dtype=float)

    return BaselineForecast(
        cycle=target_cycle,
        method="long_term_mean",
        I_total_samples=I_totals,
        metadata={"n_past_years": len(past_years), "long_term_mean": mean_I},
    )


# ---------------------------------------------------------------------------
# Skill scoring
# ---------------------------------------------------------------------------


def crps_skill_score(
    crps_model: float,
    crps_baseline: float,
    eps: float = 1e-9,
) -> float:
    """Compute CRPSS = 1 - CRPS_model / CRPS_baseline.

    * CRPSS > 0  : model beats baseline (positive skill)
    * CRPSS = 0  : model ties baseline
    * CRPSS < 0  : baseline beats model (negative skill)
    * CRPSS = 1  : perfect forecast (CRPS_model = 0)

    NaN propagates if either input is NaN. If the baseline is exactly
    zero (e.g. degenerate persistence with a perfect historical cycle),
    we return NaN to flag the situation.
    """
    if not np.isfinite(crps_model) or not np.isfinite(crps_baseline):
        return float("nan")
    if abs(crps_baseline) < eps:
        return float("nan")
    return float(1.0 - crps_model / crps_baseline)


def evaluate_baselines_for_cycle(
    historical_seasons_pr: dict[int, np.ndarray],
    historical_seasons_eto: dict[int, np.ndarray],
    *,
    target_cycle: int,
    observed_pr: np.ndarray,
    observed_eto: np.ndarray,
    kc_curve: np.ndarray,
    awc: float,
    mad: float = 0.55,
    p_eff_factor: float = 0.8,
    n_simulations: int = 500,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run all three baselines and return their CRPS against the observed.

    Returns a DataFrame with one row per baseline:
        method, I_total_obs_mm, CRPS_mm, n_samples
    """
    wb_obs = daily_water_balance(
        eto=observed_eto, pr=observed_pr, kc=kc_curve,
        awc=awc, mad=mad, p_eff_factor=p_eff_factor,
    )
    I_obs = float(wb_obs.I.sum())

    baselines = [
        naive_climatology_forecast(
            historical_seasons_pr, historical_seasons_eto,
            target_cycle=target_cycle, kc_curve=kc_curve, awc=awc, mad=mad,
            p_eff_factor=p_eff_factor, n_simulations=n_simulations,
            random_seed=random_seed,
        ),
        persistence_forecast(
            historical_seasons_pr, historical_seasons_eto,
            target_cycle=target_cycle, kc_curve=kc_curve, awc=awc, mad=mad,
            p_eff_factor=p_eff_factor, n_simulations=n_simulations,
        ),
        long_term_mean_forecast(
            historical_seasons_pr, historical_seasons_eto,
            target_cycle=target_cycle, kc_curve=kc_curve, awc=awc, mad=mad,
            p_eff_factor=p_eff_factor, n_simulations=n_simulations,
        ),
    ]

    rows = []
    for bf in baselines:
        rows.append({
            "method": bf.method,
            "I_total_obs_mm": I_obs,
            "I_total_baseline_mean_mm": float(bf.I_total_samples.mean()),
            "CRPS_mm": _crps_ensemble_1d(bf.I_total_samples, I_obs),
            "n_samples": len(bf.I_total_samples),
        })
    return pd.DataFrame(rows)
