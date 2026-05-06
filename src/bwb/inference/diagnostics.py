"""MCMC diagnostics for Bayesian inference validation.

Provides functions to assess convergence and sample quality using
standard diagnostic metrics: R-hat, effective sample size, divergent
transitions, and energy diagnostics.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import arviz as az


def check_convergence(idata: az.InferenceData) -> dict:
    """Check convergence of MCMC chains using standard diagnostics.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model with posterior samples

    Returns
    -------
    dict
        Convergence status with keys:
        - 'r_hat': dict, R-hat values by variable
        - 'ess': dict, effective sample sizes
        - 'divergences': int, total divergent transitions
        - 'converged': bool, overall convergence status
    """
    summary = az.summary(idata, var_names=None)

    # Check R-hat values
    r_hat = summary["r_hat"].to_dict()
    max_r_hat = max(r_hat.values()) if r_hat else np.inf
    converged_rhat = max_r_hat < 1.05

    # Check effective sample sizes
    ess = summary["ess_bulk"].to_dict()
    min_ess = min(ess.values()) if ess else 0
    converged_ess = min_ess > 400  # Minimum recommended ESS

    # Check divergences
    divergences = idata.sample_stats.diverging.sum().item()
    max_divergences = len(idata.posterior.chain) * len(idata.posterior.draw) * 0.05
    converged_div = divergences <= max_divergences

    # Overall convergence
    converged = converged_rhat and converged_ess and converged_div

    return {
        "r_hat": r_hat,
        "ess": ess,
        "divergences": int(divergences),
        "max_r_hat": float(max_r_hat),
        "min_ess": float(min_ess),
        "converged": converged,
    }


def compute_ess(
    idata: az.InferenceData,
    var_names: Optional[list[str]] = None,
    method: str = "bulk",
) -> dict:
    """Compute effective sample size for variables.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model
    var_names : list[str], optional
        Variables to compute ESS for
    method : str
        ESS method ('bulk', 'tail', 'local')

    Returns
    -------
    dict
        ESS values by variable
    """
    if var_names is None:
        var_names = list(idata.posterior.data_vars)

    ess_func = getattr(az, f"ess_{method}")
    ess_results = {}

    for var in var_names:
        try:
            ess_val = ess_func(idata, var_names=[var])
            ess_results[var] = float(ess_val)
        except Exception:
            ess_results[var] = np.nan

    return ess_results


def compute_r_hat(
    idata: az.InferenceData,
    var_names: Optional[list[str]] = None,
) -> dict:
    """Compute R-hat (Gelman-Rubin diagnostic) for variables.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model
    var_names : list[str], optional
        Variables to compute R-hat for

    Returns
    -------
    dict
        R-hat values by variable
    """
    if var_names is None:
        var_names = list(idata.posterior.data_vars)

    rhat_results = {}

    for var in var_names:
        try:
            rhat_val = az.rhat(idata, var_names=[var])
            rhat_results[var] = float(rhat_val)
        except Exception:
            rhat_results[var] = np.nan

    return rhat_results


def check_divergences(idata: az.InferenceData) -> dict:
    """Check for divergent transitions in MCMC sampling.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model with sample_stats

    Returns
    -------
    dict
        Divergence diagnostics
    """
    diverging = idata.sample_stats.diverging
    total_points = np.prod(diverging.shape)
    n_divergences = int(diverging.sum())
    divergence_rate = n_divergences / total_points if total_points > 0 else 0

    # Get positions of divergent transitions
    div_indices = np.where(diverging.values)[0]

    return {
        "n_divergences": n_divergences,
        "total_draws": total_points,
        "divergence_rate": float(divergence_rate),
        "divergent_indices": div_indices.tolist() if len(div_indices) < 100 else div_indices[:100].tolist(),
        "has_divergences": n_divergences > 0,
    }


def check_energy(idata: az.InferenceData) -> dict:
    """Check energy diagnostic for chain mixing.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model with sample_stats

    Returns
    -------
    dict
        Energy diagnostic results
    """
    if "energy" not in idata.sample_stats:
        return {"available": False}

    energy = idata.sample_stats.energy
    energy_diff = np.diff(energy.values, axis=0)

    return {
        "available": True,
        "mean_energy": float(energy.mean()),
        "std_energy": float(energy.std()),
        "energy_diff_mean": float(np.mean(energy_diff)),
        "energy_diff_std": float(np.std(energy_diff)),
    }


def generate_diagnostic_report(idata: az.InferenceData) -> str:
    """Generate a text diagnostic report for MCMC sampling.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model

    Returns
    -------
    str
        Formatted diagnostic report
    """
    conv = check_convergence(idata)
    div = check_divergences(idata)
    energy = check_energy(idata)

    lines = [
        "=" * 60,
        "MCMC DIAGNOSTIC REPORT",
        "=" * 60,
        "",
        f"Convergence Status: {'✓ PASSED' if conv['converged'] else '✗ FAILED'}",
        f"  Max R-hat: {conv['max_r_hat']:.3f} (target: <1.05)",
        f"  Min ESS: {conv['min_ess']:.0f} (target: >400)",
        "",
        f"Divergent Transitions: {div['n_divergences']} / {div['total_draws']} ({div['divergence_rate']:.1%})",
        "",
    ]

    if energy["available"]:
        lines.extend([
            "Energy Diagnostic:",
            f"  Mean: {energy['mean_energy']:.2f}",
            f"  Std: {energy['std_energy']:.2f}",
            "",
        ])

    lines.append("=" * 60)

    return "\n".join(lines)