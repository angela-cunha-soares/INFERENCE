"""Decision rules for irrigation scheduling based on soil moisture states.

Implements the probabilistic decision rule from the manuscript:
    Irrigate if P(theta_d < theta_crit | observations) > threshold

This module provides the core decision logic that couples the Bayesian
posterior distribution with actionable irrigation thresholds.
"""

from __future__ import annotations

import numpy as np


def apply_decision_rule(
    posterior_samples: np.ndarray,
    threshold: float,
    probability_threshold: float = 0.8,
) -> dict:
    """Apply probabilistic decision rule for irrigation scheduling.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Posterior samples of soil moisture theta_d (n_samples,)
    threshold : float
        Critical soil moisture threshold (fraction of available water)
    probability_threshold : float
        Minimum probability of exceeding threshold to recommend irrigation
        (default: 0.8 = 80% confidence)

    Returns
    -------
    dict
        Decision outcome with keys:
        - 'irrigate': bool, whether to irrigate
        - 'prob_below_threshold': float, P(theta < threshold | posterior)
        - 'recommended_depth_mm': float, recommended irrigation depth
        - 'decision': str, 'irrigate' | 'wait' | 'uncertain'
    """
    if len(posterior_samples) == 0:
        return {
            "irrigate": False,
            "prob_below_threshold": np.nan,
            "recommended_depth_mm": 0.0,
            "decision": "uncertain",
        }

    # Compute probability of being below threshold
    prob_below = np.mean(posterior_samples < threshold)

    # Decision logic
    if prob_below > probability_threshold:
        decision = "irrigate"
        irrigate = True
    elif prob_below < (probability_threshold - 0.15):
        decision = "wait"
        irrigate = False
    else:
        decision = "uncertain"
        irrigate = False

    # Compute recommended depth (distance to field capacity)
    if irrigate:
        # Recommended depth = distance from current state to field capacity
        current_estimate = np.median(posterior_samples)
        recommended_depth = max(0, (threshold - current_estimate) * 100)  # mm
    else:
        recommended_depth = 0.0

    return {
        "irrigate": irrigate,
        "prob_below_threshold": float(prob_below),
        "recommended_depth_mm": float(recommended_depth),
        "decision": decision,
    }


def compute_expected_water_deficit(
    posterior_samples: np.ndarray,
    threshold: float,
) -> float:
    """Compute expected water deficit given posterior distribution.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Posterior samples of soil moisture theta_d
    threshold : float
        Critical threshold

    Returns
    -------
    float
        Expected water deficit in mm
    """
    deficits = np.maximum(0, threshold - posterior_samples)
    return float(np.mean(deficits) * 100)  # Convert to mm


def compute_risk_score(
    posterior_samples: np.ndarray,
    threshold: float,
    penalty_factor: float = 2.0,
) -> float:
    """Compute irrigation risk score based on posterior uncertainty.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Posterior samples of soil moisture
    threshold : float
        Critical threshold
    penalty_factor : float
        Multiplier for false negative cost (default: 2.0)

    Returns
    -------
    float
        Risk score (0=low risk, 1=high risk)
    """
    prob_below = np.mean(posterior_samples < threshold)
    uncertainty = np.std(posterior_samples)

    # Risk increases with probability of stress and uncertainty
    risk = prob_below * (1 + penalty_factor * uncertainty)
    return float(np.clip(risk, 0, 1))