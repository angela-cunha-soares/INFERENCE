"""MCMC sampling utilities for Bayesian inference using PyMC.

Provides wrappers around PyMC samplers with sensible defaults for
water balance model inference.
"""

from __future__ import annotations

from typing import Optional
import pymc as pm
import arviz as az


def sample_posterior(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: Optional[int] = None,
    random_seed: Optional[int] = None,
    return_inferencedata: bool = True,
    target_accept: float = 0.8,
    max_treedepth: int = 10,
) -> az.InferenceData | dict:
    """Sample from posterior distribution using NUTS sampler.

    Parameters
    ----------
    model : pm.Model
        PyMC model to sample from
    draws : int
        Number of samples per chain after tuning (default: 1000)
    tune : int
        Number of tuning draws (default: 1000)
    chains : int
        Number of parallel chains (default: 4)
    cores : int, optional
        Number of CPU cores for parallel sampling
    random_seed : int, optional
        Random seed for reproducibility
    return_inferencedata : bool
        If True, return ArviZ InferenceData; else return trace dict
    target_accept : float
        Target acceptance rate for NUTS (default: 0.8)
    max_treedepth : int
        Maximum tree depth for NUTS (default: 10)

    Returns
    -------
    az.InferenceData or dict
        Posterior samples with group structure
    """
    if cores is None:
        cores = min(chains, 4)

    sampler_kwargs = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "target_accept": target_accept,
        "max_treedepth": max_treedepth,
        "return_inferencedata": return_inferencedata,
    }

    if random_seed is not None:
        sampler_kwargs["random_seed"] = random_seed

    with model:
        idata = pm.sample(**sampler_kwargs)

    return idata


def sample_posterior_predictive(
    model: pm.Model,
    idata: az.InferenceData,
    var_names: Optional[list[str]] = None,
    draws: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> az.InferenceData:
    """Generate posterior predictive samples.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    idata : az.InferenceData
        Fitted model with posterior samples
    var_names : list[str], optional
        Variables to generate predictive samples for
    draws : int, optional
        Number of posterior draws to use (default: all)
    random_seed : int, optional
        Random seed

    Returns
    -------
    az.InferenceData
        InferenceData with posterior_predictive group added
    """
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=var_names,
            random_seed=random_seed,
        )

    return ppc


def sample_prior_predictive(
    model: pm.Model,
    var_names: Optional[list[str]] = None,
    draws: int = 500,
    random_seed: Optional[int] = None,
) -> az.InferenceData:
    """Generate prior predictive samples for model checking.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    var_names : list[str], optional
        Variables to generate prior predictive for
    draws : int
        Number of samples (default: 500)
    random_seed : int, optional
        Random seed

    Returns
    -------
    az.InferenceData
        InferenceData with prior_predictive group
    """
    with model:
        ppc = pm.sample_prior_predictive(
            var_names=var_names,
            draws=draws,
            random_seed=random_seed,
        )

    return ppc


def get_default_sampler_config() -> dict:
    """Return default sampler configuration for water balance models.

    Returns
    -------
    dict
        Default configuration for NUTS sampler
    """
    return {
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.8,
        "max_treedepth": 10,
        "init": "adapt_diag",
    }