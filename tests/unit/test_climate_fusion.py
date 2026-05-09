"""Unit tests for bwb.data.sources.climate.fusion (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bwb.data.sources.climate.fusion import (
    HIST_WEIGHTS_NASA,
    _per_row_weighted_mean,
    _resolve_weights,
    fuse_climate_sources,
)


# ---------------------------------------------------------------------------
# Synthetic source builder
# ---------------------------------------------------------------------------


def _src(value_offset: float, n: int = 5) -> pd.DataFrame:
    """A minimal source with all canonical columns set to a constant
    plus an offset, so we can verify weighted means analytically."""
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "Rs":   20.0 + value_offset,
        "u2":    2.0 + value_offset,
        "Tmax": 30.0 + value_offset,
        "Tmin": 20.0 + value_offset,
        "RH":   60.0 + value_offset,
        "pr":    1.0 + value_offset,
        "ETo":   5.0 + value_offset,
    })


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------


def test_resolve_weights_two_known_sources_split_per_var():
    w = _resolve_weights(["nasa_power", "openmeteo_archive"])
    # Rs: NASA gets 0.92, OM gets 0.08
    assert w["Rs"] == pytest.approx({"nasa_power": 0.92, "openmeteo_archive": 0.08})
    # u2: NASA gets 0.20, OM gets 0.80
    assert w["u2"] == pytest.approx({"nasa_power": 0.20, "openmeteo_archive": 0.80})


def test_resolve_weights_two_om_sources_share_om_share():
    w = _resolve_weights(["nasa_power", "openmeteo_archive", "openmeteo_forecast"])
    # OM share = 1 - 0.92 = 0.08; split equally between archive & forecast
    assert w["Rs"]["nasa_power"] == 0.92
    assert w["Rs"]["openmeteo_archive"] == pytest.approx(0.04)
    assert w["Rs"]["openmeteo_forecast"] == pytest.approx(0.04)


def test_resolve_weights_only_om_sources_full_weight():
    w = _resolve_weights(["openmeteo_archive"])
    # No NASA → OM gets the full 1.0 share
    assert w["Tmax"]["openmeteo_archive"] == 1.0


def test_resolve_weights_eto_defaults_to_50_50():
    w = _resolve_weights(["nasa_power", "openmeteo_archive"])
    assert w["ETo"]["nasa_power"] == 0.5
    assert w["ETo"]["openmeteo_archive"] == 0.5


# ---------------------------------------------------------------------------
# Row-wise weighted mean
# ---------------------------------------------------------------------------


def test_per_row_weighted_mean_full_data():
    aligned = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3),
        "Rs__nasa_power":         [20.0, 21.0, 22.0],
        "Rs__openmeteo_archive":  [10.0, 11.0, 12.0],
    })
    weights = {"Rs": {"nasa_power": 0.92, "openmeteo_archive": 0.08}}
    out = _per_row_weighted_mean(aligned, "Rs", weights)
    expected = 0.92 * np.array([20.0, 21.0, 22.0]) + 0.08 * np.array([10.0, 11.0, 12.0])
    np.testing.assert_allclose(out, expected)


def test_per_row_weighted_mean_renormalises_when_one_source_missing():
    aligned = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=2),
        "Rs__nasa_power":         [20.0, np.nan],
        "Rs__openmeteo_archive":  [10.0, 12.0],
    })
    weights = {"Rs": {"nasa_power": 0.92, "openmeteo_archive": 0.08}}
    out = _per_row_weighted_mean(aligned, "Rs", weights)
    # Row 0: full weights → 0.92*20 + 0.08*10 = 19.2
    # Row 1: NASA missing → OM share renormalised to 1.0 → 12.0
    np.testing.assert_allclose(out, [19.2, 12.0])


def test_per_row_weighted_mean_all_nan_yields_nan():
    aligned = pd.DataFrame({
        "Rs__nasa_power":        [np.nan],
        "Rs__openmeteo_archive": [np.nan],
    })
    weights = {"Rs": {"nasa_power": 0.92, "openmeteo_archive": 0.08}}
    out = _per_row_weighted_mean(aligned, "Rs", weights)
    assert np.isnan(out[0])


# ---------------------------------------------------------------------------
# End-to-end: fuse_climate_sources
# ---------------------------------------------------------------------------


def test_fuse_two_sources_recompute_eto_returns_canonical_schema():
    nasa = _src(0.0)
    archive = _src(2.0)
    fused = fuse_climate_sources(
        {"nasa_power": nasa, "openmeteo_archive": archive},
        lat=-7.5, elevation_m=285.0,
    )
    assert list(fused.columns) == ["date", "Rs", "u2", "Tmax", "Tmin",
                                    "RH", "pr", "ETo"]
    assert len(fused) == len(nasa)
    # Tmax: NASA weight 0.58, OM 0.42 → 0.58*30 + 0.42*32 = 30.84
    np.testing.assert_allclose(fused["Tmax"].iloc[0],
                                0.58 * 30.0 + 0.42 * 32.0)
    # u2: NASA 0.20, OM 0.80 → 0.20*2 + 0.80*4 = 3.6
    np.testing.assert_allclose(fused["u2"].iloc[0], 0.20 * 2.0 + 0.80 * 4.0)
    # Rs: NASA 0.92, OM 0.08 → 0.92*20 + 0.08*22 = 20.16
    np.testing.assert_allclose(fused["Rs"].iloc[0], 0.92 * 20.0 + 0.08 * 22.0)
    # ETo recomputed → not the trivial weighted mean of input ETos
    assert fused["ETo"].iloc[0] > 0
    assert fused["ETo"].iloc[0] != pytest.approx(0.5 * 5.0 + 0.5 * 7.0)


def test_fuse_falls_back_to_eto_average_when_elevation_missing():
    nasa = _src(0.0)
    archive = _src(2.0)
    fused = fuse_climate_sources(
        {"nasa_power": nasa, "openmeteo_archive": archive},
        lat=-7.5, elevation_m=None,
    )
    # ETo defaults to NASA 0.5 / OM 0.5 → 0.5*5 + 0.5*7 = 6.0
    np.testing.assert_allclose(fused["ETo"].iloc[0], 6.0)


def test_fuse_handles_partial_date_overlap():
    nasa = _src(0.0, n=5)
    archive = _src(2.0, n=5)
    archive["date"] = archive["date"] + pd.Timedelta(days=2)  # shift by 2 days
    fused = fuse_climate_sources(
        {"nasa_power": nasa, "openmeteo_archive": archive},
        lat=-7.5, elevation_m=285.0,
    )
    # Union of dates = 5 + 2 (NASA-only at front) overlap window
    # = 7 unique days. None should be all-NaN.
    assert len(fused) == 7
    assert fused.notna().all().all()


def test_fuse_rejects_empty_sources():
    with pytest.raises(ValueError):
        fuse_climate_sources({}, lat=0.0, elevation_m=0.0)


def test_fuse_rejects_dataframe_without_date():
    bad = pd.DataFrame({"Rs": [1.0]})
    with pytest.raises(ValueError):
        fuse_climate_sources({"nasa_power": bad}, lat=0.0, elevation_m=0.0)


def test_fuse_three_sources_eto_recompute_matches_single_run():
    """Adding a third source should not break recomputation."""
    nasa = _src(0.0)
    archive = _src(1.0)
    forecast = _src(0.5)
    fused = fuse_climate_sources(
        {"nasa_power": nasa,
         "openmeteo_archive": archive,
         "openmeteo_forecast": forecast},
        lat=-7.5, elevation_m=285.0,
    )
    assert len(fused) == 5
    assert (fused["ETo"] > 0).all()
    # OM share = 0.08 split across 2 sources → each gets 0.04 for Rs
    expected_rs = 0.92 * 20.0 + 0.04 * 21.0 + 0.04 * 20.5
    np.testing.assert_allclose(fused["Rs"].iloc[0], expected_rs)


def test_fuse_hist_weights_match_evaonline():
    # Sanity check: weights match EVAOnline ``HIST_WEIGHTS`` published in
    # backend/core/data_processing/climate_fusion.py.
    assert HIST_WEIGHTS_NASA["Rs"] == 0.92
    assert HIST_WEIGHTS_NASA["u2"] == 0.20
    assert HIST_WEIGHTS_NASA["RH"] == 0.35
    assert HIST_WEIGHTS_NASA["Tmax"] == 0.58
    assert HIST_WEIGHTS_NASA["Tmin"] == 0.52
    assert HIST_WEIGHTS_NASA["pr"] == 0.50
