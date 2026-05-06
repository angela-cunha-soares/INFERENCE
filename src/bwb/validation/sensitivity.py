"""Sensitivity analysis for Bayesian water balance model parameters.

Implements Sobol' sensitivity analysis and other global sensitivity
methods to identify which parameters have the most influence on
model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    n_samples: int
    n_bootstrap: int
    parameter_ranges: dict
    output_var: str


def default_sensitivity_config() -> SensitivityConfig:
    """Default configuration for sensitivity analysis."""
    return SensitivityConfig(
        n_samples=1000,
        n_bootstrap=100,
        parameter_ranges={
            "theta_s": (0.35, 0.50),
            "theta_r": (0.05, 0.15),
            "Kc_mult": (0.85, 1.15),
            "theta_init": (0.20, 0.35),
            "sigma_obs": (0.01, 0.05),
        },
        output_var="yield",
    )


def generate_saltelli_samples(
    config: SensitivityConfig,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate Saltelli sample matrix for Sobol' analysis.

    Parameters
    ----------
    config : SensitivityConfig
        Sensitivity configuration
    random_seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Sample matrix with parameters as columns
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_params = len(config.parameter_ranges)
    n_samples = config.n_samples

    param_names = list(config.parameter_ranges.keys())
    ranges = list(config.parameter_ranges.values())

    A = np.random.uniform(
        [r[0] for r in ranges],
        [r[1] for r in ranges],
        (n_samples, n_params),
    )

    B = np.random.uniform(
        [r[0] for r in ranges],
        [r[1] for r in ranges],
        (n_samples, n_params),
    )

    n_ab = n_params * n_samples
    AB = np.zeros((n_ab, n_params))

    for i, param_idx in enumerate(range(n_params)):
        for j in range(n_samples):
            row = i * n_samples + j
            AB[row, :] = A[j, :].copy()
            AB[row, param_idx] = B[j, param_idx]

    samples = np.vstack([A, B, AB])
    df = pd.DataFrame(samples, columns=param_names)

    return df


def compute_sobol_indices(
    samples: pd.DataFrame,
    outputs: np.ndarray,
    config: SensitivityConfig,
) -> dict:
    """Compute Sobol' sensitivity indices.

    Parameters
    ----------
    samples : pd.DataFrame
        Input parameter samples
    outputs : np.ndarray
        Model outputs corresponding to samples
    config : SensitivityConfig
        Sensitivity configuration

    Returns
    -------
    dict
        First and total order Sobol' indices
    """
    n_samples = config.n_samples
    param_names = samples.columns

    A_out = outputs[:n_samples]
    B_out = outputs[n_samples:2 * n_samples]

    var_y = float(np.var(np.concatenate([A_out, B_out])))

    first_order: dict = {}
    total_order: dict = {}

    for i, param in enumerate(param_names):
        ab_start = 2 * n_samples + i * n_samples
        ab_end = ab_start + n_samples
        AB_out = outputs[ab_start:ab_end]

        if var_y > 0:
            # Saltelli (2010) estimators
            s_first = float(np.mean(B_out * (AB_out - A_out)) / var_y)
            s_total = float(0.5 * np.mean((A_out - AB_out) ** 2) / var_y)
        else:
            s_first = 0.0
            s_total = 0.0

        first_order[param] = s_first
        total_order[param] = s_total

    return {
        "first_order": first_order,
        "total_order": total_order,
        "var_y": var_y,
    }


def run_sensitivity_analysis(
    model_func,
    config: Optional[SensitivityConfig] = None,
    random_seed: Optional[int] = None,
) -> dict:
    """Run complete sensitivity analysis.

    Parameters
    ----------
    model_func : callable
        Function that takes parameter dict and returns scalar output
    config : SensitivityConfig, optional
        Configuration (uses default if None)
    random_seed : int, optional
        Random seed

    Returns
    -------
    dict
        Sensitivity indices and diagnostics
    """
    if config is None:
        config = default_sensitivity_config()

    samples = generate_saltelli_samples(config, random_seed)

    outputs_list = []
    for _, row in samples.iterrows():
        outputs_list.append(float(model_func(row.to_dict())))
    outputs = np.asarray(outputs_list, dtype=float)

    indices = compute_sobol_indices(samples, outputs, config)

    return {
        "indices": indices,
        "samples": samples,
        "outputs": outputs,
        "config": config,
    }


def bootstrap_confidence_intervals(
    samples: pd.DataFrame,
    outputs: np.ndarray,
    config: SensitivityConfig,
    n_bootstrap: int = 100,
    random_seed: Optional[int] = None,
) -> dict:
    """Compute bootstrap confidence intervals for sensitivity indices.

    Parameters
    ----------
    samples : pd.DataFrame
        Input samples
    outputs : np.ndarray
        Model outputs
    config : SensitivityConfig
        Configuration
    n_bootstrap : int
        Number of bootstrap samples
    random_seed : int, optional
        Random seed

    Returns
    -------
    dict
        Confidence intervals for each index
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(outputs)
    param_names = samples.columns

    first_order_ci = {p: [] for p in param_names}
    total_order_ci = {p: [] for p in param_names}

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_samples = samples.iloc[idx].reset_index(drop=True)
        boot_outputs = outputs[idx]

        indices = compute_sobol_indices(boot_samples, boot_outputs, config)

        for p in param_names:
            first_order_ci[p].append(indices["first_order"][p])
            total_order_ci[p].append(indices["total_order"][p])

    ci_results = {}
    for p in param_names:
        ci_results[p] = {
            "first_order": (
                float(np.percentile(first_order_ci[p], 2.5)),
                float(np.percentile(first_order_ci[p], 97.5)),
            ),
            "total_order": (
                float(np.percentile(total_order_ci[p], 2.5)),
                float(np.percentile(total_order_ci[p], 97.5)),
            ),
        }

    return ci_results


def save_sensitivity_report(
    results: dict,
    output_path,
) -> None:
    """Save sensitivity analysis report (CSV with first/total order indices)."""
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = results["indices"]
    rows = []
    for param in indices["first_order"].keys():
        rows.append({
            "parameter": param,
            "first_order": indices["first_order"][param],
            "total_order": indices["total_order"][param],
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
