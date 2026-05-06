"""Unit tests for bwb.decision.utility."""

from __future__ import annotations

import numpy as np

from bwb.decision.utility import (
    apply_decision_rule, compute_expected_water_deficit, compute_risk_score,
)


def test_decision_irrigates_when_distribution_is_below_threshold():
    samples = np.full(1000, 0.05)
    out = apply_decision_rule(samples, threshold=0.10, probability_threshold=0.8)
    assert out["irrigate"] is True
    assert out["decision"] == "irrigate"
    assert out["prob_below_threshold"] == 1.0


def test_decision_waits_when_distribution_is_above_threshold():
    samples = np.full(1000, 0.30)
    out = apply_decision_rule(samples, threshold=0.10)
    assert out["irrigate"] is False
    assert out["decision"] == "wait"


def test_decision_uncertain_in_grey_zone():
    # Decision logic: irrigate if p>0.8, wait if p<0.65, uncertain in (0.65, 0.8]
    # Build samples with ~72% below threshold to land squarely in the grey zone.
    samples = np.concatenate([np.full(720, 0.05), np.full(280, 0.30)])
    out = apply_decision_rule(samples, threshold=0.10, probability_threshold=0.8)
    assert out["decision"] == "uncertain"
    assert out["irrigate"] is False


def test_empty_samples_handled():
    out = apply_decision_rule(np.array([]), threshold=0.10)
    assert out["irrigate"] is False
    assert out["decision"] == "uncertain"


def test_expected_deficit_zero_when_above_threshold():
    samples = np.array([0.20, 0.25, 0.30])
    assert compute_expected_water_deficit(samples, 0.10) == 0.0


def test_risk_score_in_unit_interval():
    rng = np.random.default_rng(1)
    samples = rng.uniform(0, 0.5, size=200)
    score = compute_risk_score(samples, 0.10)
    assert 0.0 <= score <= 1.0
