"""Unit tests for the prior modules (Van Genuchten reference + smoke-test)."""

from __future__ import annotations

import numpy as np
import pytest

from bwb.models.van_genuchten import (
    SoilParameters, get_soil_parameters, water_retention, inverse_retention,
    compute_field_capacity, compute_wilting_point,
    compute_available_water_capacity, generate_retention_curve,
)


def test_default_soils_available():
    for name in ("sandy_loam", "loam", "clay_loam"):
        sp = get_soil_parameters(name)
        assert isinstance(sp, SoilParameters)
        assert 0.0 < sp.theta_r < sp.theta_s < 1.0


def test_water_retention_monotone_decreasing():
    sp = get_soil_parameters("loam")
    h = np.logspace(0, 4, 50)
    theta = water_retention(h, sp)
    assert np.all(np.diff(theta) <= 1e-9)
    assert theta.max() <= sp.theta_s + 1e-9
    assert theta.min() >= sp.theta_r - 1e-9


def test_water_retention_inverse_roundtrip():
    sp = get_soil_parameters("sandy_loam")
    h = np.logspace(1, 3, 20)
    theta = water_retention(h, sp)
    h_back = inverse_retention(theta, sp)
    assert np.allclose(h_back, h, rtol=1e-3, atol=1e-3)


def test_field_capacity_above_wilting_point():
    sp = get_soil_parameters("loam")
    fc = compute_field_capacity(sp)
    wp = compute_wilting_point(sp)
    assert fc > wp


def test_awc_positive():
    sp = get_soil_parameters("loam")
    assert compute_available_water_capacity(sp, root_depth_cm=60) > 0


def test_generate_retention_curve_shape():
    sp = get_soil_parameters("clay_loam")
    h, theta = generate_retention_curve(sp, n_points=50)
    assert len(h) == 50 and len(theta) == 50


def test_climatological_priors_smoke(matopiba_profile):
    """Loading the bundled MATOPIBA profile should expose crop+soil blocks."""
    assert "crop" in matopiba_profile
    assert "soil" in matopiba_profile
    assert matopiba_profile["crop"]["awc_mm"] > 0
