"""Unit tests for bwb.forecast.climatological."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bwb.forecast.climatological import (
    CLASS_NAMES,
    DirichletPosterior,
    classify_seasons,
    compute_seasonal_spei,
    extract_historical_seasons,
    forecast_cycle,
    run_sequential_forecast,
    update_dirichlet,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic 60-year daily climate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_history():
    """60 years of daily climate with three regimes (dry/normal/wet)."""
    rng = np.random.default_rng(42)
    rows = []
    for year in range(1961, 2025):
        # 3-year cycle of regimes for testing purposes
        regime = year % 3
        if regime == 0:        # dry
            p_scale, eto_scale = 0.4, 1.2
        elif regime == 1:      # normal
            p_scale, eto_scale = 1.0, 1.0
        else:                  # wet
            p_scale, eto_scale = 1.6, 0.9

        # Simulate Dec-Mar (~120 days) so the 90-day cycle starting Dec 1 fits
        start = pd.Timestamp(year=year, month=12, day=1)
        dates = pd.date_range(start, periods=120, freq="D")
        p = rng.gamma(1.5, 8.0 * p_scale, 120) * (rng.uniform(size=120) > 0.5)
        eto = np.clip(rng.normal(4.5 * eto_scale, 0.8, 120), 2.0, None)
        for d, pp, ee in zip(dates, p, eto):
            rows.append({"date": d, "pr": pp, "ETo": ee})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def soybean_profile():
    return {
        "crop": {
            "cycle_days": 90,
            "planting_month": 12,
            "planting_day": 1,
            "awc_mm": 120.0,
            "mad": 0.55,
            "kc_ini": 0.40, "kc_mid": 1.15, "kc_end": 0.50,
            "L_ini": 15, "L_dev": 15, "L_mid": 40, "L_late": 20,
            "root_depth_cm": 60,
        },
        "soil": {"theta_s": 0.43, "theta_r": 0.045},
    }


# ---------------------------------------------------------------------------
# SPEI standardisation
# ---------------------------------------------------------------------------


def test_spei_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    d = rng.normal(-50, 100, size=200)
    spei = compute_seasonal_spei(d)
    assert abs(spei.mean()) < 0.15
    assert 0.7 < spei.std() < 1.3


def test_spei_handles_negative_d():
    """D for arid regions is mostly negative; SPEI must still work."""
    d = np.array([-200.0, -150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0])
    spei = compute_seasonal_spei(d)
    assert np.all(np.isfinite(spei))
    # Monotonic: larger D -> larger (less negative) SPEI
    assert np.all(np.diff(spei) >= -1e-9)


def test_spei_too_few_seasons_raises():
    with pytest.raises(ValueError):
        compute_seasonal_spei(np.array([10.0, 20.0]))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_seasons_terciles_balanced():
    df = pd.DataFrame({
        "year": np.arange(30),
        "P_total": np.linspace(100, 600, 30),
        "ETo_total": np.full(30, 350.0),
    })
    classes, meta = classify_seasons(df, method="tercile")
    counts = {c: list(classes.values()).count(c) for c in range(3)}
    # Equal terciles should give 10/10/10
    assert counts[0] == 10
    assert counts[1] == 10
    assert counts[2] == 10
    assert meta["method"] == "tercile"


def test_classify_seasons_spei_method_returns_three_classes():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "year": np.arange(40),
        "P_total": rng.uniform(200, 600, 40),
        "ETo_total": rng.uniform(300, 450, 40),
    })
    classes, meta = classify_seasons(df, method="spei")
    assert set(classes.values()).issubset({0, 1, 2})
    assert all(np.isfinite(meta["stats_df"]["spei"]))


# ---------------------------------------------------------------------------
# Cycle extraction
# ---------------------------------------------------------------------------


def test_extract_complete_cycles_only(synthetic_history):
    stats, daily = extract_historical_seasons(
        synthetic_history, planting_month=12, planting_day=1, cycle_days=90,
    )
    assert len(stats) > 50
    for _, row in stats.iterrows():
        assert row["P_total"] > 0
        assert daily[int(row["year"])]["pr"].shape == (90,)


def test_extract_until_year_exclusive(synthetic_history):
    stats, _ = extract_historical_seasons(
        synthetic_history, until_year_exclusive=2020,
    )
    assert int(stats["year"].max()) < 2020


# ---------------------------------------------------------------------------
# Conjugate Dirichlet update
# ---------------------------------------------------------------------------


def test_dirichlet_conjugate_update():
    posterior = update_dirichlet(
        alpha_prior=np.array([1.0, 1.0, 1.0]),
        counts=np.array([5, 10, 15]),
    )
    np.testing.assert_allclose(posterior.alpha, [6, 11, 16])
    assert posterior.n_observed == 30


def test_dirichlet_expected_weights_sum_to_one():
    posterior = DirichletPosterior(
        alpha=np.array([2.0, 3.0, 5.0]), counts=np.array([1, 2, 4]), n_observed=7,
    )
    w = posterior.expected_weights()
    assert pytest.approx(w.sum()) == 1.0


def test_dirichlet_sampling_shape_and_simplex():
    posterior = DirichletPosterior(
        alpha=np.array([2.0, 3.0, 5.0]), counts=np.array([0, 0, 0]), n_observed=0,
    )
    samples = posterior.sample_weights(100, random_seed=0)
    assert samples.shape == (100, 3)
    np.testing.assert_allclose(samples.sum(axis=1), 1.0, atol=1e-9)
    assert (samples >= 0).all()


# ---------------------------------------------------------------------------
# Forecast for one cycle
# ---------------------------------------------------------------------------


def test_forecast_cycle_shapes(synthetic_history, soybean_profile):
    forecast = forecast_cycle(
        synthetic_history, target_year=2020,
        profile=soybean_profile,
        n_simulations=50, random_seed=0,
    )
    assert forecast.cycle == 2020
    assert forecast.season_label == "2020/2021"
    assert forecast.SW.shape == (50, 90)
    assert forecast.I.shape == (50, 90)
    assert forecast.sampled_classes.shape == (50,)
    assert set(forecast.sampled_classes.tolist()).issubset({0, 1, 2})


def test_forecast_cycle_quantiles_are_ordered(synthetic_history, soybean_profile):
    forecast = forecast_cycle(
        synthetic_history, target_year=2021,
        profile=soybean_profile,
        n_simulations=80, random_seed=1,
    )
    q05 = np.quantile(forecast.SW, 0.05, axis=0)
    q50 = np.quantile(forecast.SW, 0.50, axis=0)
    q95 = np.quantile(forecast.SW, 0.95, axis=0)
    assert (q05 <= q50 + 1e-9).all()
    assert (q50 <= q95 + 1e-9).all()


# ---------------------------------------------------------------------------
# Sequential forecast
# ---------------------------------------------------------------------------


def test_sequential_forecast_alpha_grows(synthetic_history, soybean_profile):
    result = run_sequential_forecast(
        synthetic_history,
        target_years=[2020, 2021, 2022],
        profile=soybean_profile,
        city="Synthetic",
        n_simulations=30,
        random_seed=0,
    )
    # alpha trajectory: 3 entries (one per cycle) showing the prior used
    assert len(result.alpha_trajectory) == 3
    np.testing.assert_allclose(result.alpha_trajectory[0], [1, 1, 1])
    # final posterior must have grown by the number of observed cycles
    n_obs = len(result.evaluations)
    assert pytest.approx(result.final_posterior.alpha.sum() - 3) == n_obs


def test_sequential_forecast_evaluations_have_metrics(synthetic_history, soybean_profile):
    result = run_sequential_forecast(
        synthetic_history,
        target_years=[2020, 2021],
        profile=soybean_profile,
        city="Synthetic",
        n_simulations=30,
        random_seed=0,
    )
    for ev in result.evaluations.values():
        assert "KGE_SW" in ev.metrics_deterministic
        assert "CRPS_I_total_mm" in ev.metrics_probabilistic
        assert ev.observed_class in {0, 1, 2}
