"""Climatological Bayesian forecast for crop water balance.

Implements the design described in the manuscript:

1. Climatological training (1961 -> cycle_target - 1):
   * Extract every complete crop cycle of cycle_days starting at the planting
     date.
   * Compute per-cycle SPEI (Vicente-Serrano et al. 2010) on the seasonal
     P-ETo balance via log-logistic + N(0,1).
   * Classify each historical season into {dry, normal, wet} using the SPEI
     terciles of the training set (Pinkayan 1966; Xavier et al. 2002).

2. Bayesian Dirichlet-Multinomial on the class counts:
   * Conjugate prior  Dir(alpha_0)
   * Counts (n_dry, n_normal, n_wet)
   * Posterior        Dir(alpha_0 + counts)
   * Posterior of cycle t becomes the prior of cycle t+1 (sequential update).

3. Forecast for the target cycle:
   * Sample weights w ~ Dir(alpha_post)
   * Sample category c ~ Categorical(w)
   * Resample one historical season of category c (preserves dry-spells and
     intra-season correlations -- the AR(1) structure is implicit).
   * Run the FAO-56 daily water balance with the resampled (P, ETo).
   * Aggregate over n_simulations Monte-Carlo draws -> distribution of
     theta(t) and irrigation(t).

4. Validation against the real cycle (Xavier reanalysis P, ETo):
   * Compute the deterministic FAO-56 trajectory for the observed cycle.
   * Compare forecast distribution vs observed.
   * Update Dirichlet alpha with the observed class -> prior of next cycle.

This is the *correct* validation design: priors strictly use 1961-(c-1), the
target cycle is never seen by the model during training of cycle c.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisk, norm

from bwb.models.fao56 import WaterBalanceResult, daily_water_balance


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ("dry", "normal", "wet")
N_CLASSES = 3


# ---------------------------------------------------------------------------
# Per-cycle statistics + classification
# ---------------------------------------------------------------------------


@dataclass
class CycleStats:
    """Aggregate statistics for a single historical crop cycle."""

    year: int
    P_total: float
    ETo_total: float
    D: float                # P - ETo (mm)
    spei: float             # Standardised P-ETo Index (Vicente-Serrano 2010)
    aridity_index: float    # P / ETo (UNEP 1992)
    klass: int              # 0=dry, 1=normal, 2=wet


@dataclass
class HistoricalSeason:
    """Daily climate forcings for a single historical crop cycle."""

    year: int
    klass: int
    P_daily: np.ndarray      # mm/day, shape (cycle_days,)
    ETo_daily: np.ndarray    # mm/day, shape (cycle_days,)
    stats: CycleStats


def compute_seasonal_spei(d_series: np.ndarray) -> np.ndarray:
    """Compute SPEI from a series of seasonal D = P - ETo.

    Procedure (Vicente-Serrano et al. 2010):
        1. Shift D so all values are positive (log-logistic requires support > 0)
        2. Fit a log-logistic (Fisk) distribution by MLE
        3. SPEI = N^{-1}(F_LL(D))

    Parameters
    ----------
    d_series : array (n_seasons,)
        Seasonal D = P_total - ETo_total per crop cycle.

    Returns
    -------
    np.ndarray of shape (n_seasons,)
        SPEI values; NaN-free; mean ~ 0, std ~ 1.
    """
    d = np.asarray(d_series, dtype=float)
    if d.size < 3:
        raise ValueError("Need at least 3 seasons to fit SPEI")

    shift = 0.0
    if d.min() <= 0:
        shift = abs(d.min()) + 1.0
    d_pos = d + shift

    params = fisk.fit(d_pos)
    cdf_values = np.clip(fisk.cdf(d_pos, *params), 1e-6, 1 - 1e-6)
    return norm.ppf(cdf_values)


def classify_seasons(
    stats_df: pd.DataFrame,
    *,
    method: str = "spei",
    low_q: float = 1.0 / 3,
    high_q: float = 2.0 / 3,
) -> tuple[dict[int, int], dict]:
    """Classify each historical season into {dry, normal, wet}.

    Parameters
    ----------
    stats_df : DataFrame with columns 'year', 'P_total', 'ETo_total'
    method : {'spei', 'tercile'}
        ``'spei'`` (default) computes SPEI per season then partitions by
        terciles. ``'tercile'`` partitions D = P - ETo directly by terciles.
    low_q, high_q : float
        Quantile thresholds (default 1/3 and 2/3 -> equal terciles).

    Returns
    -------
    classes : dict[year -> class_idx]
    metadata : dict with keys 'thresholds', 'method', 'aridity_index_mean'
    """
    df = stats_df.copy()
    df["D"] = df["P_total"] - df["ETo_total"]
    df["aridity_index"] = df["P_total"] / df["ETo_total"]

    if method == "spei":
        df["spei"] = compute_seasonal_spei(df["D"].to_numpy())
        score = df["spei"]
    elif method == "tercile":
        df["spei"] = np.nan
        score = df["D"]
    else:
        raise ValueError(f"unknown method: {method!r}")

    q_lo = float(score.quantile(low_q))
    q_hi = float(score.quantile(high_q))

    classes: dict[int, int] = {}
    for _, row in df.iterrows():
        v = score.loc[row.name]
        if v <= q_lo:
            klass = 0
        elif v <= q_hi:
            klass = 1
        else:
            klass = 2
        classes[int(row["year"])] = klass

    return classes, {
        "method": method,
        "thresholds": {"low": q_lo, "high": q_hi},
        "aridity_index_mean": float(df["aridity_index"].mean()),
        "stats_df": df,
    }


# ---------------------------------------------------------------------------
# Cycle extraction
# ---------------------------------------------------------------------------


def extract_historical_seasons(
    df: pd.DataFrame,
    *,
    planting_month: int = 12,
    planting_day: int = 1,
    cycle_days: int = 90,
    until_year_exclusive: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Slice a continuous daily climate DataFrame into complete crop cycles.

    Parameters
    ----------
    df : DataFrame with columns 'date', 'pr', 'ETo'
    planting_month, planting_day : int
        Planting calendar date (default 1 December).
    cycle_days : int
        Cycle length (default 90 days for early soybean).
    until_year_exclusive : int, optional
        If given, only seasons with start year < this value are returned.
        Use for the training set (e.g. 2020 -> uses 1961..2019).

    Returns
    -------
    stats_df : DataFrame with one row per complete cycle
        Columns: year, P_total, ETo_total
    daily_by_year : dict[year -> dict('pr', 'ETo')] of np.ndarray (cycle_days,)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    years = sorted(df["date"].dt.year.unique())
    if until_year_exclusive is not None:
        years = [y for y in years if y < until_year_exclusive]

    stats_rows = []
    daily: dict[int, dict[str, np.ndarray]] = {}
    for y in years:
        start = pd.Timestamp(year=y, month=planting_month, day=planting_day)
        end = start + pd.Timedelta(days=cycle_days - 1)
        mask = (df["date"] >= start) & (df["date"] <= end)
        season = df.loc[mask]
        if len(season) != cycle_days:
            continue
        p = season["pr"].to_numpy(dtype=float)
        eto = season["ETo"].to_numpy(dtype=float)
        stats_rows.append({
            "year": y,
            "P_total": float(p.sum()),
            "ETo_total": float(eto.sum()),
        })
        daily[y] = {"pr": p, "ETo": eto}

    if not stats_rows:
        raise ValueError("No complete crop cycle found in the historical record")
    return pd.DataFrame(stats_rows), daily


# ---------------------------------------------------------------------------
# Conjugate Dirichlet-Multinomial update
# ---------------------------------------------------------------------------


@dataclass
class DirichletPosterior:
    """Conjugate Dirichlet posterior over the (dry, normal, wet) categories."""

    alpha: np.ndarray         # shape (3,)
    counts: np.ndarray        # observed counts that produced this posterior
    n_observed: int

    def sample_weights(self, n_samples: int, random_seed: Optional[int] = None) -> np.ndarray:
        """Draw n_samples weight vectors from Dir(alpha)."""
        rng = np.random.default_rng(random_seed)
        return rng.dirichlet(self.alpha, size=n_samples)

    def expected_weights(self) -> np.ndarray:
        return self.alpha / float(self.alpha.sum())


def update_dirichlet(
    alpha_prior: np.ndarray,
    counts: np.ndarray,
) -> DirichletPosterior:
    """Conjugate update: Dir(alpha) + Multinomial(n; w) -> Dir(alpha + counts)."""
    a = np.asarray(alpha_prior, dtype=float)
    c = np.asarray(counts, dtype=float)
    if a.shape != c.shape:
        raise ValueError("alpha_prior and counts must have the same shape")
    return DirichletPosterior(
        alpha=a + c,
        counts=c.astype(int),
        n_observed=int(c.sum()),
    )


# ---------------------------------------------------------------------------
# Forecast for one cycle
# ---------------------------------------------------------------------------


@dataclass
class CycleForecast:
    """Forecast distribution for a single crop cycle."""

    cycle: int                              # planting year (cycle = year/year+1)
    season_label: str                       # e.g. "2020/2021"
    n_simulations: int
    cycle_days: int
    SW: np.ndarray                          # (n_sim, cycle_days), mm
    I: np.ndarray                           # (n_sim, cycle_days), mm/day
    DP: np.ndarray                          # (n_sim, cycle_days), mm/day
    ETc: np.ndarray                         # (n_sim, cycle_days), mm/day
    sampled_classes: np.ndarray             # (n_sim,), 0/1/2
    sampled_years: np.ndarray               # (n_sim,)
    posterior: DirichletPosterior           # before observing target cycle
    classification: dict                    # {'thresholds', 'method', ...}

    # Aggregated season totals
    @property
    def I_total(self) -> np.ndarray:
        return self.I.sum(axis=1)

    @property
    def n_irrigation_events(self) -> np.ndarray:
        return (self.I > 0).sum(axis=1)


def _resample_season(
    historical_seasons: dict[int, HistoricalSeason],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample category and then a historical year. Returns (klass, year)."""
    klass = int(rng.choice(N_CLASSES, p=weights))
    seasons_in_class = [s for s in historical_seasons.values() if s.klass == klass]
    if not seasons_in_class:
        # Fallback: resample uniformly across all classes
        seasons_in_class = list(historical_seasons.values())
    pick = rng.integers(0, len(seasons_in_class))
    return klass, seasons_in_class[pick].year


def forecast_cycle(
    historical_df: pd.DataFrame,
    target_year: int,
    *,
    profile: dict,
    alpha_prior: Optional[np.ndarray] = None,
    method: str = "spei",
    n_simulations: int = 500,
    random_seed: Optional[int] = None,
    city: str = "",
) -> CycleForecast:
    """Forecast a single crop cycle using climatological resampling.

    Parameters
    ----------
    historical_df : DataFrame with columns 'date', 'pr', 'ETo'
        Must cover at least 1961..(target_year - 1) for the training set.
    target_year : int
        Planting year of the target cycle (e.g. 2020 for the 2020/2021 cycle).
    profile : dict
        Regional profile (see bwb.config.profiles.load_profile). Must expose
        crop block: cycle_days, planting_month, planting_day, awc_mm, mad,
        kc_ini/kc_mid/kc_end, L_ini/L_dev/L_mid/L_late.
    alpha_prior : array (3,), optional
        Dirichlet prior. Defaults to uniform Dir(1, 1, 1).
    method : {'spei', 'tercile'}
        Classification method.
    n_simulations : int
        Number of Monte-Carlo draws (default 500).
    random_seed : int, optional
    """
    from bwb.data.adapters import build_kc_curve, get_awc_for_city

    crop = profile["crop"]
    cycle_days = int(crop["cycle_days"])
    planting_month = int(crop["planting_month"])
    planting_day = int(crop["planting_day"])
    awc = get_awc_for_city(profile, city)
    mad = float(crop.get("mad", 0.55))

    if alpha_prior is None:
        alpha_prior = np.ones(N_CLASSES, dtype=float)

    # 1) Training set: 1961..(target_year - 1)
    stats_df, daily = extract_historical_seasons(
        historical_df,
        planting_month=planting_month,
        planting_day=planting_day,
        cycle_days=cycle_days,
        until_year_exclusive=target_year,
    )

    # 2) Classify and count
    classes, meta = classify_seasons(stats_df, method=method)
    counts = np.array([
        sum(1 for k in classes.values() if k == c) for c in range(N_CLASSES)
    ], dtype=float)

    # 3) Conjugate Dirichlet update
    posterior = update_dirichlet(alpha_prior, counts)

    # 4) Build HistoricalSeason objects
    aug_stats = meta["stats_df"].set_index("year")
    historical_seasons: dict[int, HistoricalSeason] = {}
    for year, klass in classes.items():
        row = aug_stats.loc[year]
        cs = CycleStats(
            year=year,
            P_total=float(row["P_total"]),
            ETo_total=float(row["ETo_total"]),
            D=float(row["D"]),
            spei=float(row["spei"]),
            aridity_index=float(row["aridity_index"]),
            klass=klass,
        )
        historical_seasons[year] = HistoricalSeason(
            year=year,
            klass=klass,
            P_daily=daily[year]["pr"],
            ETo_daily=daily[year]["ETo"],
            stats=cs,
        )

    # 5) Build Kc curve
    kc = build_kc_curve(profile, n_days=cycle_days)

    # 6) Monte-Carlo forecast
    rng = np.random.default_rng(random_seed)
    weights_samples = posterior.sample_weights(n_simulations, random_seed=random_seed)

    SW_arr = np.empty((n_simulations, cycle_days), dtype=float)
    I_arr = np.empty_like(SW_arr)
    DP_arr = np.empty_like(SW_arr)
    ETc_arr = np.empty_like(SW_arr)
    sampled_classes = np.empty(n_simulations, dtype=int)
    sampled_years = np.empty(n_simulations, dtype=int)

    for i in range(n_simulations):
        klass, year = _resample_season(historical_seasons, weights_samples[i], rng)
        season = historical_seasons[year]
        wb = daily_water_balance(
            eto=season.ETo_daily, pr=season.P_daily, kc=kc,
            awc=awc, mad=mad,
        )
        SW_arr[i] = wb.SW
        I_arr[i] = wb.I
        DP_arr[i] = wb.DP
        ETc_arr[i] = wb.ETc
        sampled_classes[i] = klass
        sampled_years[i] = year

    return CycleForecast(
        cycle=target_year,
        season_label=f"{target_year}/{target_year + 1}",
        n_simulations=n_simulations,
        cycle_days=cycle_days,
        SW=SW_arr, I=I_arr, DP=DP_arr, ETc=ETc_arr,
        sampled_classes=sampled_classes,
        sampled_years=sampled_years,
        posterior=posterior,
        classification=meta,
    )


# ---------------------------------------------------------------------------
# Evaluation against the observed cycle
# ---------------------------------------------------------------------------


@dataclass
class CycleEvaluation:
    """Skill scores of a forecast against the observed cycle."""

    cycle: int
    season_label: str
    observed_class: Optional[int]      # if observable; otherwise None
    SW_obs: np.ndarray                 # (cycle_days,)
    I_obs: np.ndarray                  # (cycle_days,)
    ETc_obs: np.ndarray                # (cycle_days,)
    metrics_deterministic: dict
    metrics_probabilistic: dict
    metrics_baseline: dict = field(default_factory=dict)
    crpss: dict = field(default_factory=dict)   # CRPS Skill Score per baseline
    pit_sw_daily: Optional[np.ndarray] = None   # (cycle_days,) PIT of SW


def _crps_ensemble_1d(samples: np.ndarray, observation: float) -> float:
    """Hersbach-2000 estimator for a single observation."""
    s = np.asarray(samples, dtype=float)
    n = len(s)
    if n == 0:
        return float("nan")
    sorted_s = np.sort(s)
    weights = (2 * np.arange(1, n + 1) - n - 1) / (n ** 2)
    return float(np.mean(np.abs(s - observation)) - np.sum(weights * sorted_s))


def evaluate_against_observed(
    forecast: CycleForecast,
    observed_df: pd.DataFrame,
    profile: dict,
    *,
    classification_meta: Optional[dict] = None,
    city: str = "",
) -> CycleEvaluation:
    """Compare a CycleForecast against the realised cycle from Xavier.

    Parameters
    ----------
    forecast : CycleForecast
    observed_df : DataFrame for the target cycle (cycle_days rows, columns
        'pr', 'ETo'). Typically the rows of the city CSV between Dec 1 of
        target_year and Mar 1 of target_year + 1.
    profile : dict
        Same regional profile used for the forecast.
    classification_meta : dict, optional
        Output of classify_seasons() on the training set; used to assign a
        class to the observed cycle (re-using the training-set thresholds).
    """
    from bwb.data.adapters import build_kc_curve, get_awc_for_city

    crop = profile["crop"]
    awc = get_awc_for_city(profile, city)
    mad = float(crop.get("mad", 0.55))
    kc = build_kc_curve(profile, n_days=int(crop["cycle_days"]))

    p = observed_df["pr"].to_numpy(dtype=float)
    eto = observed_df["ETo"].to_numpy(dtype=float)
    wb_obs = daily_water_balance(eto=eto, pr=p, kc=kc, awc=awc, mad=mad)

    # Deterministic comparison: per-day forecast median / mean vs observed
    sw_med = np.median(forecast.SW, axis=0)
    i_med = np.mean(forecast.I, axis=0)

    abs_err_sw = np.abs(sw_med - wb_obs.SW)
    abs_err_i = np.abs(i_med - wb_obs.I)

    # Pearson + KGE on SW (less sparse than I)
    if np.std(wb_obs.SW) > 1e-6:
        r = float(np.corrcoef(sw_med, wb_obs.SW)[0, 1])
        alpha = float(np.std(sw_med) / np.std(wb_obs.SW))
        beta = float(np.mean(sw_med) / max(np.mean(wb_obs.SW), 1e-9))
        kge = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
        nse = 1.0 - float(np.sum((sw_med - wb_obs.SW) ** 2) /
                          np.sum((wb_obs.SW - np.mean(wb_obs.SW)) ** 2))
    else:
        r = kge = nse = float("nan")

    metrics_det = {
        "I_total_obs_mm": float(wb_obs.I.sum()),
        "I_total_forecast_median_mm": float(np.median(forecast.I_total)),
        "I_total_forecast_mean_mm": float(forecast.I_total.mean()),
        "I_total_error_mm": float(np.median(forecast.I_total) - wb_obs.I.sum()),
        "n_irrigation_events_obs": int((wb_obs.I > 0).sum()),
        "n_irrigation_events_forecast_mean": float(forecast.n_irrigation_events.mean()),
        "MAE_SW_mm": float(np.mean(abs_err_sw)),
        "MAE_I_mm": float(np.mean(abs_err_i)),
        "Pearson_r_SW": r,
        "KGE_SW": kge,
        "NSE_SW": nse,
    }

    # Probabilistic
    i_total_samples = forecast.I_total
    i_total_obs = float(wb_obs.I.sum())
    q05, q50, q95 = np.quantile(i_total_samples, [0.05, 0.5, 0.95])
    crps_total = _crps_ensemble_1d(i_total_samples, i_total_obs)
    pit_total = float(np.mean(i_total_samples <= i_total_obs))
    coverage_90 = float(q05 <= i_total_obs <= q95)

    # Per-day daily coverage of SW + per-day PIT
    sw_q05 = np.quantile(forecast.SW, 0.05, axis=0)
    sw_q95 = np.quantile(forecast.SW, 0.95, axis=0)
    coverage_daily_sw = float(np.mean(
        (wb_obs.SW >= sw_q05) & (wb_obs.SW <= sw_q95)
    ))

    # PIT per day for SW (used for reliability diagrams)
    n_members = forecast.SW.shape[0]
    pit_sw_daily = np.zeros(forecast.cycle_days, dtype=float)
    for d in range(forecast.cycle_days):
        less = float((forecast.SW[:, d] < wb_obs.SW[d]).sum())
        equal = float((forecast.SW[:, d] == wb_obs.SW[d]).sum())
        pit_sw_daily[d] = (less + 0.5 * equal) / n_members

    metrics_prob = {
        "CRPS_I_total_mm": crps_total,
        "PIT_I_total": pit_total,
        "coverage_90_I_total": coverage_90,
        "coverage_90_SW_daily": coverage_daily_sw,
        "I_total_q05": float(q05),
        "I_total_q50": float(q50),
        "I_total_q95": float(q95),
    }

    # Observed class (using training thresholds if provided)
    observed_class: Optional[int] = None
    if classification_meta is not None:
        if classification_meta["method"] == "spei":
            d_obs = float(p.sum() - eto.sum())
            # We don't refit SPEI on a single point; use D directly relative
            # to the training-set D thresholds for class assignment.
            stats_train = classification_meta["stats_df"]
            d_train = stats_train["D"].to_numpy()
            q_lo = float(np.quantile(d_train, 1.0 / 3))
            q_hi = float(np.quantile(d_train, 2.0 / 3))
        else:
            d_obs = float(p.sum() - eto.sum())
            q_lo = classification_meta["thresholds"]["low"]
            q_hi = classification_meta["thresholds"]["high"]
        if d_obs <= q_lo:
            observed_class = 0
        elif d_obs <= q_hi:
            observed_class = 1
        else:
            observed_class = 2

    return CycleEvaluation(
        cycle=forecast.cycle,
        season_label=forecast.season_label,
        observed_class=observed_class,
        SW_obs=wb_obs.SW,
        I_obs=wb_obs.I,
        ETc_obs=wb_obs.ETc,
        metrics_deterministic=metrics_det,
        metrics_probabilistic=metrics_prob,
    )


# ---------------------------------------------------------------------------
# Sequential pipeline (the main public entry point)
# ---------------------------------------------------------------------------


@dataclass
class SequentialResult:
    """Container with all forecasts, evaluations and posterior trajectory."""

    city: str
    target_years: list[int]
    forecasts: dict[int, CycleForecast] = field(default_factory=dict)
    evaluations: dict[int, CycleEvaluation] = field(default_factory=dict)
    alpha_trajectory: list[np.ndarray] = field(default_factory=list)
    final_posterior: Optional[DirichletPosterior] = None


def run_sequential_forecast(
    historical_df: pd.DataFrame,
    target_years: Iterable[int],
    *,
    profile: dict,
    city: str = "",
    alpha_init: Optional[np.ndarray] = None,
    method: str = "spei",
    n_simulations: int = 500,
    random_seed: Optional[int] = 42,
    on_cycle_done=None,
) -> SequentialResult:
    """Run the sequential forecast loop over consecutive crop cycles.

    For each cycle c in target_years:
        forecast(c) using all data in historical_df with year < c
        evaluate(c) against the realised cycle (if data available)
        update alpha with the observed class -> prior of cycle c+1

    Parameters
    ----------
    historical_df : DataFrame with columns 'date', 'pr', 'ETo'
        Must cover at least 1961..max(target_years).
    target_years : iterable of int
        Planting years to forecast (e.g. [2020, 2021, 2022, 2023, 2024]).
    profile : dict
        Regional profile.
    city : str
        Annotation only.
    alpha_init : array (3,), optional
        Initial Dirichlet prior (default uniform Dir(1, 1, 1)).
    method : {'spei', 'tercile'}
    n_simulations : int
    random_seed : int, optional
    on_cycle_done : callable(year, forecast, evaluation), optional

    Returns
    -------
    SequentialResult
    """
    if alpha_init is None:
        alpha_init = np.ones(N_CLASSES, dtype=float)

    target_years = list(target_years)
    result = SequentialResult(city=city, target_years=target_years)

    alpha = np.asarray(alpha_init, dtype=float).copy()
    crop = profile["crop"]
    cycle_days = int(crop["cycle_days"])
    planting_month = int(crop["planting_month"])
    planting_day = int(crop["planting_day"])

    historical_df = historical_df.copy()
    historical_df["date"] = pd.to_datetime(historical_df["date"])

    for c in target_years:
        result.alpha_trajectory.append(alpha.copy())

        forecast = forecast_cycle(
            historical_df, c,
            profile=profile,
            alpha_prior=alpha,
            method=method,
            n_simulations=n_simulations,
            random_seed=(random_seed + c) if random_seed is not None else None,
            city=city,
        )
        result.forecasts[c] = forecast

        # Try to evaluate against the realised cycle
        start = pd.Timestamp(year=c, month=planting_month, day=planting_day)
        end = start + pd.Timedelta(days=cycle_days - 1)
        observed = historical_df[
            (historical_df["date"] >= start) & (historical_df["date"] <= end)
        ]
        evaluation: Optional[CycleEvaluation] = None
        if len(observed) == cycle_days:
            evaluation = evaluate_against_observed(
                forecast, observed, profile,
                classification_meta=forecast.classification,
                city=city,
            )

            # Compute baselines and CRPS skill scores
            try:
                from bwb.data.adapters import build_kc_curve
                from bwb.forecast.baselines import (
                    crps_skill_score, evaluate_baselines_for_cycle,
                )
                hist_stats, hist_daily = extract_historical_seasons(
                    historical_df,
                    planting_month=planting_month,
                    planting_day=planting_day,
                    cycle_days=cycle_days,
                    until_year_exclusive=c,
                )
                hist_pr = {y: hist_daily[y]["pr"] for y in hist_daily}
                hist_eto = {y: hist_daily[y]["ETo"] for y in hist_daily}
                kc = build_kc_curve(profile, n_days=cycle_days)
                obs_pr = observed["pr"].to_numpy(dtype=float)
                obs_eto = observed["ETo"].to_numpy(dtype=float)
                bdf = evaluate_baselines_for_cycle(
                    hist_pr, hist_eto,
                    target_cycle=c,
                    observed_pr=obs_pr,
                    observed_eto=obs_eto,
                    kc_curve=kc,
                    awc=get_awc_for_city(profile, city),
                    mad=float(profile["crop"].get("mad", 0.55)),
                    n_simulations=n_simulations,
                    random_seed=(random_seed + c) if random_seed is not None else None,
                )
                evaluation.metrics_baseline = {
                    f"CRPS_{r['method']}_mm": float(r["CRPS_mm"])
                    for _, r in bdf.iterrows()
                }
                crps_model = float(evaluation.metrics_probabilistic["CRPS_I_total_mm"])
                evaluation.crpss = {
                    f"CRPSS_vs_{r['method']}": crps_skill_score(
                        crps_model, float(r["CRPS_mm"]),
                    )
                    for _, r in bdf.iterrows()
                }
            except Exception as exc:  # noqa: BLE001 - keep pipeline alive
                evaluation.metrics_baseline = {"_error": str(exc)}
                evaluation.crpss = {}

            result.evaluations[c] = evaluation

            # Sequential update: alpha -> alpha + e_observed_class
            if evaluation.observed_class is not None:
                inc = np.zeros(N_CLASSES)
                inc[evaluation.observed_class] = 1.0
                alpha = alpha + inc

        if on_cycle_done is not None:
            on_cycle_done(c, forecast, evaluation)

    result.final_posterior = DirichletPosterior(
        alpha=alpha,
        counts=alpha - np.asarray(alpha_init, dtype=float),
        n_observed=int((alpha - np.asarray(alpha_init, dtype=float)).sum()),
    )
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def forecast_to_dataframe(forecast: CycleForecast, kc: np.ndarray) -> pd.DataFrame:
    """Convert a CycleForecast to a tidy per-day DataFrame for CSV output."""
    days = np.arange(1, forecast.cycle_days + 1)
    return pd.DataFrame({
        "dia_ciclo": days,
        "Kc": kc,
        "ETc_media": forecast.ETc.mean(axis=0),
        "SW_media": forecast.SW.mean(axis=0),
        "SW_q05": np.quantile(forecast.SW, 0.05, axis=0),
        "SW_q50": np.quantile(forecast.SW, 0.50, axis=0),
        "SW_q95": np.quantile(forecast.SW, 0.95, axis=0),
        "I_media": forecast.I.mean(axis=0),
        "I_q05": np.quantile(forecast.I, 0.05, axis=0),
        "I_q50": np.quantile(forecast.I, 0.50, axis=0),
        "I_q95": np.quantile(forecast.I, 0.95, axis=0),
        "Prob_irrig": (forecast.I > 0).mean(axis=0),
    })


def save_sequential_outputs(
    result: SequentialResult,
    output_dir: Path,
    profile: dict,
) -> dict:
    """Write per-cycle CSVs, summary tables and a manifest to output_dir."""
    from bwb.data.adapters import build_kc_curve

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop = profile["crop"]
    kc = build_kc_curve(profile, n_days=int(crop["cycle_days"]))

    written: dict[str, list[str]] = {"per_cycle": [], "summary": []}

    for year, forecast in result.forecasts.items():
        df = forecast_to_dataframe(forecast, kc)
        path = output_dir / f"forecast_{result.city or 'city'}_{year}_{year + 1}.csv"
        df.to_csv(path, index=False)
        written["per_cycle"].append(str(path))

        if year in result.evaluations:
            ev = result.evaluations[year]
            obs_df = pd.DataFrame({
                "dia_ciclo": np.arange(1, forecast.cycle_days + 1),
                "SW_obs": ev.SW_obs,
                "I_obs": ev.I_obs,
                "ETc_obs": ev.ETc_obs,
            })
            obs_path = output_dir / f"observed_{result.city or 'city'}_{year}_{year + 1}.csv"
            obs_df.to_csv(obs_path, index=False)
            written["per_cycle"].append(str(obs_path))

    # Summary metrics table
    rows = []
    for year, forecast in result.forecasts.items():
        ev = result.evaluations.get(year)
        row = {
            "city": result.city,
            "cycle": forecast.season_label,
            "n_simulations": forecast.n_simulations,
            "alpha_dry": float(forecast.posterior.alpha[0]),
            "alpha_normal": float(forecast.posterior.alpha[1]),
            "alpha_wet": float(forecast.posterior.alpha[2]),
            "p_dry": float(forecast.posterior.expected_weights()[0]),
            "p_normal": float(forecast.posterior.expected_weights()[1]),
            "p_wet": float(forecast.posterior.expected_weights()[2]),
        }
        if ev is not None:
            row["observed_class"] = (
                CLASS_NAMES[ev.observed_class] if ev.observed_class is not None else "n/a"
            )
            row.update({f"det_{k}": v for k, v in ev.metrics_deterministic.items()})
            row.update({f"prob_{k}": v for k, v in ev.metrics_probabilistic.items()})
            row.update({f"baseline_{k}": v for k, v in ev.metrics_baseline.items()})
            row.update(ev.crpss)
        rows.append(row)

    summary_path = output_dir / f"sequential_summary_{result.city or 'city'}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    written["summary"].append(str(summary_path))

    return written
