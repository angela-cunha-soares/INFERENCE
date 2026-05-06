"""Unit tests for bwb.validation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from bwb.validation import metrics as M


def test_perfect_match():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert M.rmse(y, y) == pytest.approx(0.0)
    assert M.mae(y, y) == pytest.approx(0.0)
    assert M.bias(y, y) == pytest.approx(0.0)
    assert M.nse(y, y) == pytest.approx(1.0)
    assert M.kge(y, y) == pytest.approx(1.0)


def test_handles_nan_pairs():
    sim = np.array([1.0, np.nan, 3.0, 4.0])
    obs = np.array([1.0, 2.0, np.nan, 4.0])
    assert M.rmse(sim, obs) == pytest.approx(0.0)


def test_kge_decomposes_bias():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = obs + 1.0  # constant bias of +1
    val = M.kge(sim, obs)
    assert val < 1.0
    assert val > -1.0


def test_crps_perfect_concentration_is_zero():
    obs = np.array([1.0, 2.0, 3.0])
    fs = np.tile(obs[:, None], (1, 50))
    assert M.crps_mean(fs, obs) == pytest.approx(0.0, abs=1e-9)


def test_crps_against_known_value():
    """For a single ensemble {0,1} and obs=0, CRPS = 0.25 exactly."""
    crps = M.crps_ensemble(np.array([[0.0, 1.0]]), np.array([0.0]))
    assert float(crps[0]) == pytest.approx(0.25)


def test_pit_uniform_for_well_calibrated_forecast():
    rng = np.random.default_rng(0)
    n_t = 500
    obs = rng.normal(size=n_t)
    fs = rng.normal(size=(n_t, 200))
    pit = M.pit(fs, obs)
    assert pit.min() >= 0
    assert pit.max() <= 1
    # mean should be near 0.5 for well-calibrated
    assert abs(pit.mean() - 0.5) < 0.1


def test_coverage_and_interval_score():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    lo = obs - 0.5
    hi = obs + 0.5
    assert M.coverage(lo, hi, obs) == 1.0
    assert M.interval_score(lo, hi, obs, alpha=0.1) > 0


def test_compute_all_metrics_keys():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = obs + 0.1
    fs = np.tile(sim[:, None], (1, 100))
    out = M.compute_all_metrics(sim, obs, fs)
    for k in ("rmse", "mae", "bias", "pbias", "kge", "nse",
              "crps_mean", "pit_alpha", "coverage_90", "interval_score_90"):
        assert k in out
        assert np.isfinite(out[k])
