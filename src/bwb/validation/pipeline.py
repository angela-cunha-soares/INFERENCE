"""Validation pipeline for the Bayesian water balance framework.

Orchestrates a grid of (city × soil-depth × crop-cycle) combinations, runs the
Bayesian model on each one, compares the posterior mean trajectory against the
deterministic FAO-56 baseline, and reports both deterministic (KGE/RMSE/...)
and probabilistic (CRPS/PIT/coverage) metrics.

Designed to be called both from :mod:`scripts.run_validation` and from the
CLI (``python -m bwb.cli validate``).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CITIES = [
    "Baixa_Grande_do_Ribeiro", "Balsas", "Barreiras", "Bom_Jesus",
    "Campos_Lindos", "Correntina", "Formosa_do_Rio_Preto",
    "Luis_Eduardo_Magalhaes", "Tasso_Fragoso", "Urucui",
]

SOIL_DEPTHS_CM = [40, 60, 80]
CROP_CYCLES = [2020, 2021, 2022, 2023, 2024]

N_SAMPLES = 2000
N_SIMULATIONS = 500


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ValidationConfig:
    """Configuration for a validation run."""
    cities: list[str]
    soil_depths: list[int]
    crop_cycles: list[int]
    n_samples: int
    n_simulations: int
    output_dir: Path
    profile_name: str = "matopiba"
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    target_accept: float = 0.95
    random_seed: int = 42
    fast_mode: bool = False


def create_validation_config(
    output_dir: Path,
    cities: Optional[list[str]] = None,
    soil_depths: Optional[list[int]] = None,
    crop_cycles: Optional[list[int]] = None,
    *,
    profile_name: str = "matopiba",
    fast_mode: bool = False,
    draws: Optional[int] = None,
    tune: Optional[int] = None,
    chains: Optional[int] = None,
) -> ValidationConfig:
    """Create a ValidationConfig with sensible defaults.

    The default ``chains=1`` keeps the pipeline efficient on Windows where
    PyMC's multiprocess spawn dominates the runtime for short cycles.
    Convergence is still checked per-combination via diagnostics.
    """
    return ValidationConfig(
        cities=cities or CITIES,
        soil_depths=soil_depths or SOIL_DEPTHS_CM,
        crop_cycles=crop_cycles or CROP_CYCLES,
        n_samples=N_SAMPLES,
        n_simulations=N_SIMULATIONS,
        output_dir=Path(output_dir),
        profile_name=profile_name,
        draws=draws if draws is not None else (300 if fast_mode else 800),
        tune=tune if tune is not None else (300 if fast_mode else 800),
        chains=chains if chains is not None else 1,
        fast_mode=fast_mode,
    )


def generate_validation_grid(config: ValidationConfig) -> pd.DataFrame:
    """Cartesian product of cities × soil_depths × crop_cycles."""
    rows = [
        {"city": city, "soil_depth_cm": depth, "crop_cycle": cycle}
        for city in config.cities
        for depth in config.soil_depths
        for cycle in config.crop_cycles
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-combination runner
# ---------------------------------------------------------------------------


def run_single_validation(
    city: str,
    soil_depth_cm: int,
    crop_cycle: int,
    config: ValidationConfig,
    *,
    profile: Optional[dict] = None,
) -> dict:
    """Run a Bayesian water-balance validation for a single combination.

    Returns a dict with config + diagnostics + metrics.
    """
    from bwb.config.profiles import load_profile
    from bwb.data.adapters import build_water_balance_inputs, cycle_label
    from bwb.data.loaders import extract_crop_cycle, load_city_series
    from bwb.models.water_balance import fit_model, diagnostics_summary
    from bwb.validation.metrics import compute_all_metrics

    if profile is None:
        profile = load_profile(config.profile_name)

    # Override root depth from grid value
    profile = {**profile, "crop": {**profile.get("crop", {}), "root_depth_cm": soil_depth_cm}}

    base_record = {
        "city": city,
        "soil_depth_cm": soil_depth_cm,
        "crop_cycle": crop_cycle,
        "season": cycle_label(crop_cycle),
    }

    try:
        df = load_city_series(city)
    except FileNotFoundError as e:
        return {**base_record, "status": "missing_data", "error": str(e)}

    try:
        cycle_df = extract_crop_cycle(
            df,
            crop_cycle,
            planting_month=int(profile["crop"]["planting_month"]),
            planting_day=int(profile["crop"]["planting_day"]),
            cycle_days=int(profile["crop"]["cycle_days"]),
        )
    except ValueError as e:
        return {**base_record, "status": "incomplete_cycle", "error": str(e)}

    inputs = build_water_balance_inputs(
        cycle_df, profile, city=city, season=base_record["season"],
    )

    try:
        idata, _ = fit_model(
            inputs,
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            progressbar=False,
        )
    except Exception as e:
        return {
            **base_record,
            "status": "sampling_failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    diag = diagnostics_summary(idata)

    # Posterior mean trajectory (deterministic point forecast)
    theta_post = idata.posterior["theta"].values  # (chain, draw, day)
    theta_flat = theta_post.reshape(-1, theta_post.shape[-1])
    sim_mean = theta_flat.mean(axis=0)

    # Probabilistic metrics use the posterior PREDICTIVE distribution of the
    # observation (theta + sigma_obs noise). Using only the latent theta would
    # produce artificially narrow credible intervals and under-coverage.
    if "posterior_predictive" in idata and "theta_obs" in idata.posterior_predictive:
        ppc = idata.posterior_predictive["theta_obs"].values  # (chain, draw, day)
        ppc_flat = ppc.reshape(-1, ppc.shape[-1])
        forecast_samples = ppc_flat.T  # (n_days, n_draws)
    else:
        forecast_samples = theta_flat.T

    metrics = compute_all_metrics(
        simulated=sim_mean,
        observed=inputs.theta_observed,
        forecast_samples=forecast_samples,
        interval_alpha=0.1,
    )

    return {
        **base_record,
        "status": "ok",
        **{f"diag_{k}": v for k, v in diag.items()},
        **{f"metric_{k}": v for k, v in metrics.items()},
    }


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def run_validation_pipeline(
    config: ValidationConfig,
    *,
    parallel: bool = False,
    n_jobs: int = 1,
    on_progress=None,
) -> pd.DataFrame:
    """Run the full validation pipeline.

    Note: ``parallel`` is honoured only if joblib is installed; otherwise the
    pipeline falls back to a sequential loop.
    """
    grid = generate_validation_grid(config)
    total = len(grid)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if parallel and n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            from bwb.config.profiles import load_profile
            profile = load_profile(config.profile_name)
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(run_single_validation)(
                    row["city"], row["soil_depth_cm"], row["crop_cycle"],
                    config, profile=profile,
                )
                for _, row in grid.iterrows()
            )
        except ImportError:
            parallel = False

    if not parallel or n_jobs <= 1:
        results = []
        for idx, row in grid.iterrows():
            result = run_single_validation(
                row["city"], row["soil_depth_cm"], row["crop_cycle"], config,
            )
            results.append(result)
            if on_progress is not None:
                on_progress(idx + 1, total, result)
            elif (idx + 1) % 5 == 0 or (idx + 1) == total:
                pct = 100.0 * (idx + 1) / total
                print(f"  [{idx + 1:3d}/{total}] ({pct:5.1f}%) {row['city']} "
                      f"{row['soil_depth_cm']}cm {row['crop_cycle']} -> {result.get('status')}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Persistence + reporting
# ---------------------------------------------------------------------------


def save_validation_results(
    results: pd.DataFrame,
    output_dir: Path,
    prefix: str = "validation",
) -> dict:
    """Save validation results to CSV + Parquet + JSON manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    parquet_path = output_dir / f"{prefix}_{timestamp}.parquet"

    # Drop high-cardinality 'traceback' before saving parquet (keeps file lean)
    save_df = results.drop(columns=["traceback"], errors="ignore")
    save_df.to_csv(csv_path, index=False)
    try:
        save_df.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None  # type: ignore[assignment]

    manifest = {
        "csv": str(csv_path),
        "parquet": str(parquet_path) if parquet_path else None,
        "n_combinations": int(len(results)),
        "n_ok": int((results.get("status") == "ok").sum()),
        "timestamp": timestamp,
    }
    (output_dir / f"{prefix}_{timestamp}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest


def generate_validation_report(results: pd.DataFrame) -> str:
    """Human-readable text report from validation results."""
    lines = [
        "=" * 72,
        " BAYESIAN WATER BALANCE -- VALIDATION REPORT",
        "=" * 72,
        "",
        f"Total combinations : {len(results)}",
    ]

    if "status" in results.columns:
        counts = results["status"].value_counts()
        lines.append("Status breakdown   :")
        for status, count in counts.items():
            lines.append(f"  {status:<20s} : {count}")
        lines.append("")

    metric_cols = [c for c in results.columns if c.startswith("metric_")]
    ok = results[results.get("status") == "ok"] if "status" in results.columns else results
    if metric_cols and len(ok) > 0:
        lines.append("Metric summary (mean across successful runs):")
        for col in metric_cols:
            vals = pd.to_numeric(ok[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            lines.append(
                f"  {col[len('metric_'):]:<22s}  mean={vals.mean():>8.3f}  "
                f"median={vals.median():>8.3f}  std={vals.std():>8.3f}"
            )

    lines.append("=" * 72)
    return "\n".join(lines)


def to_summary_dict(config: ValidationConfig) -> dict:
    """Serialise a config to a JSON-friendly dict."""
    d = asdict(config)
    d["output_dir"] = str(d["output_dir"])
    return d
