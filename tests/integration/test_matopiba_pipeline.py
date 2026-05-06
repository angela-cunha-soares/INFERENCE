"""Integration tests that exercise the full data → inputs → ensemble path.

These tests deliberately avoid running NUTS sampling (which is slow); the
heavy MCMC is exercised separately by ``run_validation.py --fast``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bwb.data.adapters import build_water_balance_inputs, run_deterministic_baseline
from bwb.data.loaders import (
    extract_crop_cycle, list_available_cities, load_balsas_historical,
)
from bwb.forecast.ensemble import (
    PosteriorParameters, propagate_ensemble, ensemble_quantiles,
    decision_from_ensemble, synthetic_perturbed_ensemble,
)
from bwb.phenology.kc_curves import soybean_kc_90d
from bwb.validation.metrics import compute_all_metrics
from bwb.validation.pipeline import (
    create_validation_config, generate_validation_grid,
)


@pytest.mark.skipif(
    not list_available_cities(),
    reason="No city CSVs available — run scripts/extract_xavier_matopiba.py first.",
)
def test_load_and_build_inputs(matopiba_profile):
    df = load_balsas_historical()
    cycle_df = extract_crop_cycle(df, 2023)
    inputs = build_water_balance_inputs(cycle_df, matopiba_profile,
                                         city="Balsas", season="2023/2024")
    assert len(inputs.P_daily) == 90
    assert len(inputs.theta_observed) == 90
    assert inputs.soil_params["theta_s"] > inputs.soil_params["theta_r"]


def test_deterministic_baseline_clipped_to_soil_range():
    rng = np.random.default_rng(0)
    n = 90
    p = rng.gamma(1.5, 8.0, n)
    eto = np.clip(rng.normal(4.5, 0.8, n), 2.0, None)
    kc = soybean_kc_90d()
    theta = run_deterministic_baseline(p, eto, kc)
    assert theta.min() >= 0.10 - 1e-9
    assert theta.max() <= 0.45 + 1e-9


def test_ensemble_propagation_shapes_and_calibration():
    rng = np.random.default_rng(1)
    n_days, n_members, n_draws = 90, 31, 100
    p_mean = rng.gamma(1.5, 8.0, n_days)
    eto_mean = np.clip(rng.normal(4.5, 0.8, n_days), 2.0, None)
    P_ens, ETo_ens = synthetic_perturbed_ensemble(p_mean, eto_mean,
                                                  n_members=n_members,
                                                  random_seed=2)
    posterior = PosteriorParameters(
        theta_s=rng.normal(0.45, 0.02, n_draws),
        theta_r=rng.normal(0.10, 0.01, n_draws),
        Kc_mult=rng.normal(1.0, 0.05, n_draws),
        theta_init=rng.normal(0.275, 0.02, n_draws),
    )
    traj = propagate_ensemble(P_ens, ETo_ens, soybean_kc_90d(), posterior)
    assert traj.shape == (n_days, n_members, n_draws)
    qs = ensemble_quantiles(traj)
    assert qs["q05"].shape == (n_days,)
    assert np.all(qs["q05"] <= qs["q50"])
    assert np.all(qs["q50"] <= qs["q95"])
    dec = decision_from_ensemble(traj, theta_crit=0.12)
    assert set(np.unique(dec["irrigate"])).issubset({0, 1})


def test_compute_all_metrics_runs_on_pipeline_outputs():
    rng = np.random.default_rng(42)
    obs = rng.normal(0.3, 0.05, 90)
    sim = obs + rng.normal(0, 0.01, 90)
    fs = sim[:, None] + rng.normal(0, 0.02, (90, 100))
    out = compute_all_metrics(sim, obs, fs)
    assert out["rmse"] > 0
    assert out["kge"] > 0
    assert 0 <= out["coverage_90"] <= 1


def test_grid_size_for_default_config(tmp_path):
    config = create_validation_config(tmp_path / "out")
    grid = generate_validation_grid(config)
    assert len(grid) == 10 * 3 * 5
