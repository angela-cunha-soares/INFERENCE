"""Unit tests for bwb.models.fao56 (daily water balance in mm)."""

from __future__ import annotations

import numpy as np
import pytest

from bwb.models.fao56 import WaterBalanceResult, daily_water_balance


def _flat_inputs(n: int = 90, eto_val: float = 4.0, p_val: float = 5.0,
                 kc_val: float = 1.0):
    return (
        np.full(n, eto_val, dtype=float),
        np.full(n, p_val, dtype=float),
        np.full(n, kc_val, dtype=float),
    )


def test_returns_arrays_of_correct_length():
    eto, pr, kc = _flat_inputs(n=90)
    result = daily_water_balance(eto, pr, kc, awc=120.0)
    assert isinstance(result, WaterBalanceResult)
    assert result.SW.shape == (90,)
    assert result.I.shape == (90,)
    assert result.DP.shape == (90,)
    assert result.ETc.shape == (90,)


def test_sw_stays_within_zero_and_awc():
    rng = np.random.default_rng(0)
    n = 90
    eto = rng.uniform(2, 6, n)
    pr = rng.gamma(1.5, 8.0, n) * (rng.uniform(size=n) > 0.5)
    kc = rng.uniform(0.4, 1.2, n)
    out = daily_water_balance(eto, pr, kc, awc=120.0)
    assert (out.SW >= 0).all()
    assert (out.SW <= 120.0 + 1e-9).all()


def test_etc_equals_kc_times_eto():
    eto, pr, kc = _flat_inputs(eto_val=5.0, kc_val=0.8)
    out = daily_water_balance(eto, pr, kc, awc=120.0)
    np.testing.assert_allclose(out.ETc, 0.8 * 5.0)


def test_no_irrigation_when_rainfall_meets_demand():
    """P_eff = 0.8 * 6 = 4.8 > ETc = 1.0 * 4 = 4 -> SW filled, no irrigation."""
    eto, pr, kc = _flat_inputs(eto_val=4.0, p_val=6.0, kc_val=1.0)
    out = daily_water_balance(eto, pr, kc, awc=120.0)
    assert out.I.sum() == 0.0


def test_irrigation_triggers_when_below_threshold():
    """ETc=5, P=0 -> SW depletes ~5 mm/day; should trigger irrigation."""
    eto, pr, kc = _flat_inputs(eto_val=5.0, p_val=0.0, kc_val=1.0)
    out = daily_water_balance(eto, pr, kc, awc=120.0, mad=0.55)
    assert out.I.sum() > 0
    # Whenever irrigation fires, SW must equal AWC immediately after
    irrig_days = out.I > 0
    assert np.allclose(out.SW[irrig_days], 120.0)


def test_initial_soil_water_default_is_full_capacity():
    eto, pr, kc = _flat_inputs(eto_val=0.0, p_val=0.0)  # no flux
    out = daily_water_balance(eto, pr, kc, awc=120.0)
    # Day 0: SW = AWC + 0 - 0 = 120; threshold = 120 * (1-0.55) = 54;
    # 120 > 54 so no irrigation, SW stays at 120.
    np.testing.assert_allclose(out.SW, 120.0)


def test_explicit_initial_soil_water():
    eto, pr, kc = _flat_inputs(eto_val=0.0, p_val=0.0)
    out = daily_water_balance(eto, pr, kc, awc=120.0, sw_init=80.0)
    # Day 0: 80 + 0 - 0 = 80, above threshold 54 -> no irrigation
    assert out.SW[0] == pytest.approx(80.0)


def test_deep_percolation_when_rainfall_exceeds_capacity():
    """P_eff_d = 0.8 * 200 = 160 > AWC -> 40 mm DP."""
    eto = np.array([0.0])
    pr = np.array([200.0])
    kc = np.array([0.0])
    out = daily_water_balance(eto, pr, kc, awc=120.0, sw_init=120.0)
    assert out.DP[0] == pytest.approx(160.0)
    assert out.SW[0] == pytest.approx(120.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        daily_water_balance(
            np.zeros(10), np.zeros(20), np.zeros(10), awc=120.0,
        )
