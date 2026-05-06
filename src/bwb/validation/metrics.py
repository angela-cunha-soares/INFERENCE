"""Deterministic and probabilistic validation metrics.

Implements the metric suite required by the Bayesian Water Balance manuscript:

Deterministic:
    - rmse, mae, bias, pbias  (Janssen & Heuberger 1995)
    - kge                     (Kling-Gupta efficiency, Gupta et al. 2009)
    - nse                     (Nash-Sutcliffe, Nash & Sutcliffe 1970)

Probabilistic:
    - crps_ensemble           (Continuous Ranked Probability Score, ensemble form)
    - crps_mean               (mean CRPS across timesteps)
    - pit                     (Probability Integral Transform, Dawid 1984)
    - pit_alpha_reliability   (alpha reliability, Renard et al. 2010)
    - coverage                (empirical coverage of credible intervals)
    - interval_score          (Gneiting & Raftery 2007)
    - compute_all_metrics     (one-shot summary)

All functions accept aligned NumPy arrays. NaN values are dropped pairwise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_nan_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------


def rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Root mean square error."""
    s, o = _drop_nan_pairs(simulated, observed)
    if s.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((s - o) ** 2)))


def mae(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Mean absolute error."""
    s, o = _drop_nan_pairs(simulated, observed)
    if s.size == 0:
        return float("nan")
    return float(np.mean(np.abs(s - o)))


def bias(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Mean bias (sim - obs)."""
    s, o = _drop_nan_pairs(simulated, observed)
    if s.size == 0:
        return float("nan")
    return float(np.mean(s - o))


def pbias(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Percent bias (%) = 100 * sum(sim - obs) / sum(obs)."""
    s, o = _drop_nan_pairs(simulated, observed)
    denom = float(np.sum(o))
    if s.size == 0 or denom == 0.0:
        return float("nan")
    return float(100.0 * np.sum(s - o) / denom)


def nse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency coefficient."""
    s, o = _drop_nan_pairs(simulated, observed)
    if s.size == 0:
        return float("nan")
    o_mean = float(np.mean(o))
    denom = float(np.sum((o - o_mean) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((s - o) ** 2) / denom)


def kge(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Kling-Gupta efficiency (Gupta et al. 2009).

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
        r     : Pearson correlation
        alpha : sigma_sim / sigma_obs
        beta  : mu_sim    / mu_obs
    """
    s, o = _drop_nan_pairs(simulated, observed)
    if s.size < 2:
        return float("nan")
    o_std, s_std = float(np.std(o, ddof=0)), float(np.std(s, ddof=0))
    o_mean, s_mean = float(np.mean(o)), float(np.mean(s))
    if o_std == 0.0 or o_mean == 0.0:
        return float("nan")
    r = float(np.corrcoef(s, o)[0, 1])
    if not np.isfinite(r):
        return float("nan")
    alpha = s_std / o_std
    beta = s_mean / o_mean
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------


def crps_ensemble(
    forecast_samples: np.ndarray,
    observation: float | np.ndarray,
) -> float | np.ndarray:
    """Continuous Ranked Probability Score from an ensemble forecast.

    Uses the empirical-CDF formulation (Hersbach 2000):
        CRPS = E|X - y| - 0.5 * E|X - X'|

    Parameters
    ----------
    forecast_samples : array of shape (n_members,) or (n_times, n_members)
        Ensemble members.
    observation : scalar or array of shape (n_times,)
        Observed value(s).

    Returns
    -------
    float or ndarray
        Per-time CRPS (or a single scalar if input is 1-D).
    """
    fs = np.asarray(forecast_samples, dtype=float)
    if fs.ndim == 1:
        fs = fs[np.newaxis, :]
        obs = np.asarray([observation], dtype=float)
        squeeze = True
    else:
        obs = np.asarray(observation, dtype=float)
        squeeze = False

    if obs.shape[0] != fs.shape[0]:
        raise ValueError(
            f"observation length {obs.shape[0]} != forecast rows {fs.shape[0]}"
        )

    n_members = fs.shape[1]
    if n_members == 0:
        return float("nan") if squeeze else np.full(obs.shape, np.nan)

    # E|X - y|
    term1 = np.mean(np.abs(fs - obs[:, None]), axis=1)

    # 0.5 * E|X - X'|  via vectorised pairwise differences
    sorted_fs = np.sort(fs, axis=1)
    weights = (2 * np.arange(1, n_members + 1) - n_members - 1) / (n_members ** 2)
    term2 = np.sum(weights[None, :] * sorted_fs, axis=1)

    crps = term1 - term2
    if squeeze:
        return float(crps[0])
    return crps


def crps_mean(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
) -> float:
    """Mean CRPS across timesteps."""
    crps_per_t = crps_ensemble(forecast_samples, observations)
    if isinstance(crps_per_t, float):
        return crps_per_t
    crps_per_t = np.asarray(crps_per_t, dtype=float)
    crps_per_t = crps_per_t[np.isfinite(crps_per_t)]
    if crps_per_t.size == 0:
        return float("nan")
    return float(np.mean(crps_per_t))


def pit(
    forecast_samples: np.ndarray,
    observation: float | np.ndarray,
) -> np.ndarray:
    """Probability Integral Transform from ensemble forecast.

    Returns u_t = F_hat(y_t) where F_hat is the empirical forecast CDF.
    Under perfect calibration, u_t ~ Uniform(0, 1).
    """
    fs = np.asarray(forecast_samples, dtype=float)
    if fs.ndim == 1:
        fs = fs[np.newaxis, :]
        obs = np.asarray([observation], dtype=float)
    else:
        obs = np.asarray(observation, dtype=float)

    if obs.shape[0] != fs.shape[0]:
        raise ValueError("observation length mismatch with forecast rows")

    n_members = fs.shape[1]
    if n_members == 0:
        return np.full(obs.shape, np.nan)

    # Empirical CDF with mid-rank tie-handling
    less = np.sum(fs < obs[:, None], axis=1)
    equal = np.sum(fs == obs[:, None], axis=1)
    pit_values = (less + 0.5 * equal) / n_members
    return pit_values


def pit_alpha_reliability(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
) -> float:
    """Alpha reliability index from PIT (Renard et al. 2010, Eq. 12).

    alpha = 1 - 2/n * sum_i |u_(i) - i/(n+1)|

    alpha = 1 indicates a perfect uniform PIT (perfectly calibrated).
    alpha = 0 indicates the worst possible calibration.
    """
    pit_values = pit(forecast_samples, observations)
    pit_values = pit_values[np.isfinite(pit_values)]
    n = pit_values.size
    if n == 0:
        return float("nan")
    sorted_pit = np.sort(pit_values)
    expected = np.arange(1, n + 1) / (n + 1)
    return float(1.0 - 2.0 / n * np.sum(np.abs(sorted_pit - expected)))


def coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Empirical coverage of a credible interval [lower, upper]."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    o = np.asarray(observed, dtype=float)
    if not (lo.shape == hi.shape == o.shape):
        raise ValueError("Shape mismatch among lower/upper/observed")
    mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(o)
    if not mask.any():
        return float("nan")
    inside = (o[mask] >= lo[mask]) & (o[mask] <= hi[mask])
    return float(np.mean(inside))


def interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Mean interval score (Gneiting & Raftery 2007, Eq. 43).

    IS_alpha(L, U; y) = (U - L) + (2/alpha)*(L - y)*1{y<L}
                                + (2/alpha)*(y - U)*1{y>U}

    Lower is better. ``alpha`` is the nominal miscoverage (e.g., 0.1 for 90% PI).
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    o = np.asarray(observed, dtype=float)
    mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(o)
    if not mask.any():
        return float("nan")
    lo, hi, o = lo[mask], hi[mask], o[mask]
    width = hi - lo
    under = (lo - o) * (o < lo)
    over = (o - hi) * (o > hi)
    score = width + (2.0 / alpha) * (under + over)
    return float(np.mean(score))


def reliability_alpha(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> dict:
    """Empirical vs. nominal coverage at multiple credibility levels."""
    fs = np.asarray(forecast_samples, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if fs.ndim == 1:
        fs = fs[np.newaxis, :]
        obs = obs.reshape(1)

    out: dict = {}
    for level in levels:
        lo_q = (1 - level) / 2
        hi_q = 1 - lo_q
        lo = np.quantile(fs, lo_q, axis=1)
        hi = np.quantile(fs, hi_q, axis=1)
        out[level] = {
            "nominal": float(level),
            "empirical": coverage(lo, hi, obs),
        }
    return out


def compute_pit(forecast_cdf, observation) -> np.ndarray:
    """Backwards-compat alias used by some scripts.

    If `forecast_cdf` is callable, it's evaluated at `observation`.
    Otherwise it is interpreted as ensemble samples.
    """
    if callable(forecast_cdf):
        return np.asarray(forecast_cdf(observation), dtype=float)
    return pit(np.asarray(forecast_cdf), observation)


def compute_crps(
    forecast_samples: np.ndarray,
    observation: float | np.ndarray,
) -> float:
    """Backwards-compat alias for :func:`crps_mean`."""
    obs = np.atleast_1d(np.asarray(observation, dtype=float))
    return crps_mean(forecast_samples, obs)


def compute_kge(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Backwards-compat alias for :func:`kge`."""
    return kge(simulated, observed)


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------


def compute_all_metrics(
    simulated: np.ndarray,
    observed: np.ndarray,
    forecast_samples: Optional[np.ndarray] = None,
    interval_alpha: float = 0.1,
) -> dict:
    """Return a dict with all available metrics.

    Parameters
    ----------
    simulated : np.ndarray
        Point forecasts (e.g., posterior mean).
    observed : np.ndarray
        Observations.
    forecast_samples : np.ndarray, optional
        Ensemble samples of shape (n_times, n_members). If provided,
        probabilistic metrics are computed.
    interval_alpha : float
        Miscoverage for the interval score (default 0.1 = 90% PI).
    """
    out = {
        "rmse": rmse(simulated, observed),
        "mae": mae(simulated, observed),
        "bias": bias(simulated, observed),
        "pbias": pbias(simulated, observed),
        "kge": kge(simulated, observed),
        "nse": nse(simulated, observed),
    }

    if forecast_samples is not None:
        fs = np.asarray(forecast_samples, dtype=float)
        if fs.ndim == 1:
            fs = fs[np.newaxis, :]
        out["crps_mean"] = crps_mean(fs, observed)
        out["pit_alpha"] = pit_alpha_reliability(fs, observed)

        lo = np.quantile(fs, interval_alpha / 2, axis=1)
        hi = np.quantile(fs, 1 - interval_alpha / 2, axis=1)
        out[f"coverage_{int((1 - interval_alpha) * 100)}"] = coverage(lo, hi, observed)
        out[f"interval_score_{int((1 - interval_alpha) * 100)}"] = interval_score(
            lo, hi, observed, alpha=interval_alpha
        )

    return out
