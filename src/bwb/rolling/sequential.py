"""Sequential Bayesian updating for rolling irrigation decisions.

Implements the sequential inference framework where posterior distributions
from previous decision cycles become priors for subsequent cycles, enabling
continuous learning across the crop season.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SequentialState:
    """State container for sequential Bayesian updates.

    Attributes
    ----------
    cycle : int
        Crop cycle number (1-indexed)
    date : datetime
        Current date in the cycle
    posterior_samples : np.ndarray
        Posterior samples of soil moisture
    posterior_params : dict
        Posterior hyperparameters for next prior
    irrigation_decisions : list
        History of irrigation decisions made
    """
    cycle: int
    date: datetime
    posterior_samples: np.ndarray
    posterior_params: dict
    irrigation_decisions: list


def initialize_sequential_state(
    cycle: int,
    initial_soil_moisture: float,
    awc_mm: float,
) -> SequentialState:
    """Initialize state for a new crop cycle.

    Parameters
    ----------
    cycle : int
        Cycle number (e.g., 2020)
    initial_soil_moisture : float
        Initial soil moisture as fraction of AWC
    awc_mm : float
        Available water capacity in mm

    Returns
    -------
    SequentialState
        Initialized state
    """
    # Use climatological prior for first update
    posterior_params = {
        "mean": initial_soil_moisture,
        "std": 0.05,
        "awc_mm": awc_mm,
    }

    return SequentialState(
        cycle=cycle,
        date=datetime.now(),
        posterior_samples=np.array([]),
        posterior_params=posterior_params,
        irrigation_decisions=[],
    )


def update_posterior_with_observations(
    prior_params: dict,
    new_observations: np.ndarray,
    likelihood_std: float = 0.03,
) -> dict:
    """Update prior parameters with new observations using conjugate update.

    For normal likelihood with normal prior, the posterior is also normal
    with updated mean and variance.

    Parameters
    ----------
    prior_params : dict
        Prior parameters with 'mean' and 'std'
    new_observations : np.ndarray
        New soil moisture observations
    likelihood_std : float
        Observation likelihood standard deviation

    Returns
    -------
    dict
        Updated posterior parameters
    """
    prior_mean = prior_params["mean"]
    prior_var = prior_params["std"] ** 2
    obs_var = likelihood_std ** 2

    n = len(new_observations)
    if n == 0:
        return prior_params

    obs_mean = np.mean(new_observations)

    # Posterior mean (precision-weighted)
    posterior_var = 1 / (1 / prior_var + n / obs_var)
    posterior_mean = posterior_var * (prior_mean / prior_var + obs_mean * n / obs_var)

    return {
        "mean": float(posterior_mean),
        "std": float(np.sqrt(posterior_var)),
        "awc_mm": prior_params.get("awc_mm", 100),
    }


def sample_from_posterior(
    posterior_params: dict,
    n_samples: int = 1000,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Generate samples from posterior distribution.

    Parameters
    ----------
    posterior_params : dict
        Posterior parameters with 'mean' and 'std'
    n_samples : int
        Number of samples
    random_seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Posterior samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    return np.random.normal(
        posterior_params["mean"],
        posterior_params["std"],
        n_samples,
    )


def compute_posterior_predictive(
    posterior_params: dict,
    n_days: int,
    process_error_std: float = 0.02,
) -> np.ndarray:
    """Generate posterior predictive samples for future days.

    Parameters
    ----------
    posterior_params : dict
        Posterior parameters
    n_days : int
        Number of days to predict
    process_error_std : float
        Process model error standard deviation

    Returns
    -------
    np.ndarray
        Predictive samples (n_days, n_samples)
    """
    n_samples = 1000
    base_samples = np.random.normal(
        posterior_params["mean"],
        posterior_params["std"],
        (n_days, n_samples),
    )

    # Add process error
    process_error = np.random.normal(0, process_error_std, (n_days, n_samples))
    predictive = base_samples + process_error

    return predictive


def rolling_update(
    state: SequentialState,
    new_observations: np.ndarray,
    decision: dict,
) -> SequentialState:
    """Perform sequential update with new observations and decision.

    Parameters
    ----------
    state : SequentialState
        Current state
    new_observations : np.ndarray
        New soil moisture observations
    decision : dict
        Decision outcome from apply_decision_rule

    Returns
    -------
    SequentialState
        Updated state
    """
    # Update posterior parameters
    updated_params = update_posterior_with_observations(
        state.posterior_params,
        new_observations,
    )

    # Generate new posterior samples
    new_samples = sample_from_posterior(updated_params)

    # Record decision
    decisions = state.irrigation_decisions + [
        {
            "date": state.date.isoformat(),
            "decision": decision.get("decision", "unknown"),
            "irrigate": decision.get("irrigate", False),
            "prob_below": decision.get("prob_below_threshold", np.nan),
        }
    ]

    return SequentialState(
        cycle=state.cycle,
        date=datetime.now(),
        posterior_samples=new_samples,
        posterior_params=updated_params,
        irrigation_decisions=decisions,
    )


def compute_seasonal_summary(state: SequentialState) -> dict:
    """Compute summary statistics for the season.

    Parameters
    ----------
    state : SequentialState
        Final state of the season

    Returns
    -------
    dict
        Seasonal summary
    """
    if not state.irrigation_decisions:
        return {"cycle": state.cycle, "n_decisions": 0}

    decisions_df = pd.DataFrame(state.irrigation_decisions)

    return {
        "cycle": state.cycle,
        "n_decisions": len(decisions_df),
        "n_irrigations": int(decisions_df["irrigate"].sum()),
        "irrigation_rate": float(decisions_df["irrigate"].mean()),
        "final_soil_moisture": float(state.posterior_params["mean"]),
    }