"""Prior identification pipeline for climate distributions.

This module provides the high-level interface for identifying prior
distributions from the distribution atlas, similar to the standalone
identify_distributions.py script but integrated into the bwb package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def load_distribution_atlas(
    atlas_path: Optional[Path] = None,
) -> dict:
    """Load the distribution atlas from processed data.

    Parameters
    ----------
    atlas_path : Path, optional
        Path to distribution atlas JSON (uses default if None)

    Returns
    -------
    dict
        Nested atlas: {city: {variable: {month: params}}}
    """
    if atlas_path is None:
        from bwb.utils.paths import project_data_dir
        atlas_path = project_data_dir() / "processed" / "distributions" / "distribution_atlas.json"

    if not atlas_path.exists():
        raise FileNotFoundError(f"Distribution atlas not found: {atlas_path}")

    import json
    with open(atlas_path) as f:
        return json.load(f)


def get_prior_for_location(
    city: str,
    variable: str,
    month: int,
    atlas: Optional[dict] = None,
) -> dict:
    """Get prior parameters for a specific location and time.

    Parameters
    ----------
    city : str
        City name
    variable : str
        Variable name (e.g., 'pr', 'ETo', 'Tmax')
    month : int
        Month (1-12)
    atlas : dict, optional
        Distribution atlas (loads default if None)

    Returns
    -------
    dict
        Prior parameters with family and fitted parameters
    """
    if atlas is None:
        atlas = load_distribution_atlas()

    if city not in atlas:
        raise ValueError(f"City not found in atlas: {city}")

    if variable not in atlas[city]:
        raise ValueError(f"Variable not found: {variable}")

    month_key = f"{month:02d}"
    if month_key not in atlas[city][variable]:
        raise ValueError(f"Month {month} not found for {variable}")

    return atlas[city][variable][month_key]


def sample_from_prior(
    prior_params: dict,
    n_samples: int,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Generate samples from a prior distribution.

    Parameters
    ----------
    prior_params : dict
        Prior parameters with 'family' and 'params'
    n_samples : int
        Number of samples
    random_seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Samples from the prior
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    family = prior_params.get("family", "norm")
    params = prior_params.get("params", {})

    if family == "norm":
        return np.random.normal(params.get("loc", 0), params.get("scale", 1), n_samples)

    elif family == "lognorm":
        return np.random.lognormal(params.get("s", 0), params.get("scale", 1), n_samples)

    elif family == "gamma":
        return np.random.gamma(params.get("a", 1), params.get("scale", 1), n_samples)

    elif family == "zigamma":
        # Zero-inflated gamma
        p_dry = params.get("p_dry", 0.5)
        samples = []
        for _ in range(n_samples):
            if np.random.random() < p_dry:
                samples.append(0)
            else:
                shape = params.get("shape", 1)
                scale = params.get("scale", 1)
                samples.append(np.random.gamma(shape, scale))
        return np.array(samples)

    elif family == "beta_rescaled":
        # Beta distribution rescaled to [0, 1]
        a = params.get("a", 1)
        b = params.get("b", 1)
        return np.random.beta(a, b, n_samples)

    elif family == "weibull_min":
        c = params.get("c", 1)
        scale = params.get("scale", 1)
        return np.random.weibull(c, n_samples) * scale

    elif family == "skewnorm":
        return stats.skewnorm.rvs(
            params.get("alpha", 0),
            loc=params.get("loc", 0),
            scale=params.get("scale", 1),
            size=n_samples,
        )

    else:
        raise ValueError(f"Unknown prior family: {family}")


def build_climatological_prior(
    city: str,
    variable: str,
    start_month: int = 1,
    end_month: int = 12,
    atlas: Optional[dict] = None,
) -> dict:
    """Build climatological prior spanning multiple months.

    Parameters
    ----------
    city : str
        City name
    variable : str
        Variable name
    start_month : int
        Start month (1-12)
    end_month : int
        End month (1-12)
    atlas : dict, optional
        Distribution atlas

    Returns
    -------
    dict
        Combined prior parameters
    """
    if atlas is None:
        atlas = load_distribution_atlas()

    # Collect parameters for each month
    monthly_params = []
    for month in range(start_month, end_month + 1):
        try:
            prior = get_prior_for_location(city, variable, month, atlas)
            monthly_params.append(prior)
        except ValueError:
            continue

    if not monthly_params:
        raise ValueError(f"No prior data found for {city}/{variable}")

    # Combine by computing mixture parameters
    # For simplicity, use mean of parameters
    combined_params = {"family": monthly_params[0]["family"], "params": {}}

    if "params" in monthly_params[0]:
        param_keys = monthly_params[0]["params"].keys()
        for key in param_keys:
            values = [p["params"].get(key, 0) for p in monthly_params if "params" in p]
            combined_params["params"][key] = float(np.mean(values))

    return combined_params


def identify_priors_for_region(
    cities: list[str],
    variables: list[str],
    months: Optional[list[int]] = None,
    atlas: Optional[dict] = None,
) -> pd.DataFrame:
    """Identify priors for a region across multiple locations.

    Parameters
    ----------
    cities : list[str]
        List of city names
    variables : list[str]
        List of variables
    months : list[int], optional
        List of months (default: 1-12)
    atlas : dict, optional
        Distribution atlas

    Returns
    -------
    pd.DataFrame
        Prior parameters for each combination
    """
    if months is None:
        months = list(range(1, 13))

    if atlas is None:
        atlas = load_distribution_atlas()

    rows = []
    for city in cities:
        for var in variables:
            for month in months:
                try:
                    prior = get_prior_for_location(city, var, month, atlas)
                    rows.append({
                        "city": city,
                        "variable": var,
                        "month": month,
                        "family": prior.get("family"),
                        **prior.get("params", {}),
                    })
                except ValueError:
                    continue

    return pd.DataFrame(rows)