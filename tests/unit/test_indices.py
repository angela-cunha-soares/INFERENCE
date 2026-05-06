"""Unit tests for bwb.priors.indices (SPEI, IA, quantile classification)."""

from __future__ import annotations

import numpy as np
import pytest

from bwb.priors.indices import (
    aridity_index, classify_aridity, classify_spei, quantile_classify,
    season_summary, spei, standardise_nonparametric, water_balance_d,
)


def test_aridity_index_units():
    p = np.array([100.0, 50.0, 0.0])
    eto = np.array([100.0, 100.0, 100.0])
    ia = aridity_index(p, eto)
    assert ia[0] == pytest.approx(1.0)
    assert ia[1] == pytest.approx(0.5)
    assert ia[2] == 0.0


def test_aridity_division_by_zero_yields_nan():
    p = np.array([100.0])
    eto = np.array([0.0])
    ia = aridity_index(p, eto)
    assert np.isnan(ia[0])


def test_aridity_classes_match_unep():
    ia = np.array([0.03, 0.10, 0.30, 0.55, 0.80])
    classes = classify_aridity(ia)
    assert list(classes) == ["hyperarid", "arid", "semi_arid", "dry_subhumid", "humid"]


def test_water_balance_d():
    p = np.array([100.0, 50.0])
    eto = np.array([60.0, 80.0])
    assert np.allclose(water_balance_d(p, eto), [40.0, -30.0])


def test_standardise_nonparametric_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 5.0, size=300)
    z = standardise_nonparametric(x)
    assert abs(np.mean(z)) < 0.05
    assert 0.9 < np.std(z) < 1.1


def test_spei_global_standardisation():
    rng = np.random.default_rng(1)
    p = rng.gamma(1.5, 80, size=240)
    eto = rng.normal(110, 15, size=240).clip(50)
    s = spei(p, eto, timescale=1)
    finite = s[~np.isnan(s)]
    assert abs(np.mean(finite)) < 0.05
    assert 0.9 < np.std(finite) < 1.1


def test_spei_per_month_stationary():
    rng = np.random.default_rng(2)
    months = np.tile(np.arange(1, 13), 30)
    # Pure seasonal cycle in P that should be removed by per-month standardisation
    p = 80 + 60 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 10, size=360)
    eto = 120 + 30 * np.cos(2 * np.pi * months / 12) + rng.normal(0, 5, size=360)
    s = spei(p, eto, timescale=1, by_month=months)
    finite = s[~np.isnan(s)]
    assert abs(np.mean(finite)) < 0.1


def test_classify_spei_severity():
    values = np.array([-2.5, -1.7, -1.2, 0.0, 1.2, 1.7, 2.5])
    classes = classify_spei(values)
    assert list(classes) == [
        "extremely_dry", "severely_dry", "moderately_dry",
        "near_normal",
        "moderately_wet", "severely_wet", "extremely_wet",
    ]


def test_quantile_classify_terciles():
    rng = np.random.default_rng(0)
    x = rng.normal(size=900)
    classes = quantile_classify(x)
    counts = dict(zip(*np.unique(classes, return_counts=True)))
    # Each tercile should get ~300 entries (+/- 30 in 900)
    for key in ("dry", "normal", "wet"):
        assert 250 <= counts[key] <= 350


def test_quantile_classify_with_external_reference():
    """Classifying current season against historical reference."""
    rng = np.random.default_rng(3)
    reference = rng.gamma(2.0, 50.0, size=500)
    series = np.array([float(np.percentile(reference, 5)),
                       float(np.percentile(reference, 50)),
                       float(np.percentile(reference, 95))])
    classes = quantile_classify(series, reference=reference)
    assert classes[0] == "dry"
    assert classes[1] == "normal"
    assert classes[2] == "wet"


def test_season_summary_keys():
    p = np.array([5.0, 0.5, 12.0, 0.0])
    eto = np.array([4.0, 5.0, 4.5, 4.5])
    summary = season_summary(p, eto)
    for key in ("P_total_mm", "ETo_total_mm", "deficit_mm",
                "aridity_index", "n_dry_days"):
        assert key in summary
    assert summary["P_total_mm"] == 17.5
    assert summary["ETo_total_mm"] == 18.0
    assert summary["aridity_index"] == pytest.approx(17.5 / 18.0)
    assert summary["n_dry_days"] == 2  # values < 1 mm
