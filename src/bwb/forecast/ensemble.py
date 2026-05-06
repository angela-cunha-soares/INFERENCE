"""Ensemble forecast propagation through the Bayesian water balance model.

Five-day probabilistic forecasts integrate the M-member meteorological
ensemble (e.g., GEFS) with the K posterior parameter draws via nested Monte
Carlo, propagating both meteorological and parametric uncertainty into the
credible intervals reported to end users.

Notation
--------
Let
    P_m, ETo_m   : 1D arrays of length n_days, member m of the meteorological ensemble
    K_m          : Kc curve for member m (typically shared across members)
    theta_k      : posterior draw k of the parameter vector
                   (theta_s, theta_r, Kc_mult, theta_init)

For every (m, k) pair we run the FAO-56 water balance recursion using
``theta_k``'s parameters and ``P_m, ETo_m``. The output is the matrix
``theta[t, m, k]`` of soil-moisture trajectories.

This module exposes the core driver :func:`propagate_ensemble`, helpers to
build ensemble matrices, and quantile / decision summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Posterior draw extraction
# ---------------------------------------------------------------------------


@dataclass
class PosteriorParameters:
    """Posterior draws of the four water-balance parameters."""

    theta_s: np.ndarray      # (n_draws,)
    theta_r: np.ndarray      # (n_draws,)
    Kc_mult: np.ndarray      # (n_draws,)
    theta_init: np.ndarray   # (n_draws,)

    def __post_init__(self):
        n = len(self.theta_s)
        for name, arr in [("theta_r", self.theta_r), ("Kc_mult", self.Kc_mult),
                          ("theta_init", self.theta_init)]:
            if len(arr) != n:
                raise ValueError(
                    f"{name} length {len(arr)} != theta_s length {n}"
                )

    @property
    def n_draws(self) -> int:
        return len(self.theta_s)

    def subsample(self, n: int, random_seed: Optional[int] = None) -> "PosteriorParameters":
        """Draw `n` random samples (with replacement)."""
        if n >= self.n_draws:
            return self
        rng = np.random.default_rng(random_seed)
        idx = rng.integers(0, self.n_draws, size=n)
        return PosteriorParameters(
            theta_s=self.theta_s[idx],
            theta_r=self.theta_r[idx],
            Kc_mult=self.Kc_mult[idx],
            theta_init=self.theta_init[idx],
        )


def posterior_from_idata(idata) -> PosteriorParameters:
    """Extract the four parameter draws from an ArviZ InferenceData."""
    post = idata.posterior
    flatten = lambda name: post[name].values.reshape(-1)
    return PosteriorParameters(
        theta_s=flatten("theta_s"),
        theta_r=flatten("theta_r"),
        Kc_mult=flatten("Kc_mult"),
        theta_init=flatten("theta_init"),
    )


# ---------------------------------------------------------------------------
# Core propagator
# ---------------------------------------------------------------------------


def propagate_ensemble(
    P_ensemble: np.ndarray,          # (n_members, n_days)
    ETo_ensemble: np.ndarray,        # (n_members, n_days)
    Kc_curve: np.ndarray,            # (n_days,)
    posterior: PosteriorParameters,
    *,
    z_r_m: float = 0.60,
    p_eff_factor: float = 0.8,
    n_posterior_draws: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Propagate a meteorological ensemble through posterior parameter draws.

    Parameters
    ----------
    P_ensemble, ETo_ensemble : array (n_members, n_days)
        Per-member daily forcings (mm/day).
    Kc_curve : array (n_days,)
        Crop coefficient curve (shared across members; per-day variability of
        Kc is captured by the posterior ``Kc_mult``).
    posterior : PosteriorParameters
        Posterior draws.
    z_r_m : float
        Effective root depth in metres.
    p_eff_factor : float
        Effective precipitation factor.
    n_posterior_draws : int, optional
        Sub-sample the posterior to this many draws (saves memory).
    random_seed : int, optional
        Seed for posterior subsampling.

    Returns
    -------
    np.ndarray of shape (n_days, n_members, n_draws)
        Soil moisture trajectories for every (member, draw) pair.
    """
    P = np.asarray(P_ensemble, dtype=float)
    ETo = np.asarray(ETo_ensemble, dtype=float)
    if P.shape != ETo.shape:
        raise ValueError(f"P/ETo shape mismatch: {P.shape} vs {ETo.shape}")

    n_members, n_days = P.shape
    if Kc_curve.shape[0] != n_days:
        raise ValueError(
            f"Kc length {Kc_curve.shape[0]} does not match n_days {n_days}"
        )

    if n_posterior_draws is not None and n_posterior_draws < posterior.n_draws:
        posterior = posterior.subsample(n_posterior_draws, random_seed=random_seed)

    n_draws = posterior.n_draws
    z_r_mm = z_r_m * 1000.0

    # State as (n_members, n_draws); broadcast theta_init across members
    theta = np.broadcast_to(posterior.theta_init[None, :], (n_members, n_draws)).copy()
    out = np.empty((n_days, n_members, n_draws), dtype=float)

    theta_s = posterior.theta_s[None, :]
    theta_r = posterior.theta_r[None, :]
    Kc_mult = posterior.Kc_mult[None, :]

    out[0] = theta
    for d in range(1, n_days):
        kc_t = Kc_curve[d]
        # P[:, d] shape (n_members,); broadcast to (n_members, 1)
        p_eff = p_eff_factor * P[:, d : d + 1]
        etc = kc_t * Kc_mult * ETo[:, d : d + 1]
        new_theta = theta + (p_eff - etc) / z_r_mm
        np.clip(new_theta, theta_r, theta_s, out=new_theta)
        theta = new_theta
        out[d] = theta

    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def ensemble_quantiles(
    trajectories: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> dict:
    """Reduce (n_days, n_members, n_draws) → per-day quantile arrays."""
    flat = trajectories.reshape(trajectories.shape[0], -1)
    return {
        f"q{int(q * 100):02d}": np.quantile(flat, q, axis=1)
        for q in quantiles
    }


def probability_below(
    trajectories: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Per-day P(theta < threshold) across the (member × draw) ensemble."""
    flat = trajectories.reshape(trajectories.shape[0], -1)
    return np.mean(flat < threshold, axis=1)


def decision_from_ensemble(
    trajectories: np.ndarray,
    theta_crit: float,
    tau_risk: float = 0.05,
) -> dict:
    """Translate ensemble trajectories into a per-day irrigation decision."""
    p_def = probability_below(trajectories, theta_crit)
    irrigate = (p_def > tau_risk).astype(int)
    return {
        "p_deficit": p_def,
        "irrigate": irrigate,
        "theta_crit": theta_crit,
        "tau_risk": tau_risk,
    }


# ---------------------------------------------------------------------------
# Synthetic ensemble for testing / fallback
# ---------------------------------------------------------------------------


def synthetic_perturbed_ensemble(
    P_mean: np.ndarray,
    ETo_mean: np.ndarray,
    n_members: int = 31,
    p_cv: float = 0.30,
    eto_cv: float = 0.10,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a multiplicative-noise ensemble around (P_mean, ETo_mean).

    Useful when no real GEFS pull is available — gives the propagator
    something to chew on for tests and offline runs.
    """
    rng = np.random.default_rng(random_seed)
    n_days = len(P_mean)
    p_factors = rng.lognormal(mean=0.0, sigma=p_cv, size=(n_members, n_days))
    eto_factors = rng.lognormal(mean=0.0, sigma=eto_cv, size=(n_members, n_days))
    return P_mean[None, :] * p_factors, ETo_mean[None, :] * eto_factors
