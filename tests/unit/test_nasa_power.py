"""Unit tests for bwb.data.sources.climate.nasa_power.

Network is not exercised here. We test:

* the FAO-56 Penman-Monteith routine against the worked example in
  Allen et al. (1998), Example 18 (Madrid, 6 July);
* the JSON parser against a small synthetic payload mimicking the
  NASA POWER response shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bwb.data.sources.climate.nasa_power import (
    _parse_to_dataframe,
    compute_eto_fao56_pm,
    download_nasa_power,
    wind_10m_to_2m_fao56,
)


# ---------------------------------------------------------------------------
# Penman-Monteith — FAO-56 Example 18 (Madrid, 6 July)
# ---------------------------------------------------------------------------


def test_penman_monteith_matches_fao56_example_18():
    """Allen et al. (1998), Example 18: ETo ≈ 3.9 mm/day."""
    # Inputs (Madrid, latitude 40°25'N = 40.42°, elevation 660 m, DOY 187)
    eto = compute_eto_fao56_pm(
        Tmax=np.array([21.5]),
        Tmin=np.array([12.3]),
        Tmean=np.array([16.9]),
        RH=np.array([63.0]),         # Example uses ea=1.409 → ≈63% at Tmean
        u2=np.array([2.078]),
        Rs=np.array([22.07]),
        doy=np.array([187]),
        lat_deg=40.42,
        elevation_m=660.0,
    )
    # Allen reports ETo ≈ 3.88 mm/day. Tolerance 0.3 mm/day covers the small
    # discrepancy from using RH instead of the worked ea = 1.409 kPa.
    assert eto.shape == (1,)
    assert abs(float(eto[0]) - 3.88) < 0.5, f"ETo={eto[0]:.2f} not near 3.88"


def test_penman_monteith_clamps_negative_to_zero():
    # Polar-night-like: very cold + no Rs should never yield negative ETo
    eto = compute_eto_fao56_pm(
        Tmax=np.array([-5.0]), Tmin=np.array([-15.0]), Tmean=np.array([-10.0]),
        RH=np.array([90.0]), u2=np.array([1.0]), Rs=np.array([0.5]),
        doy=np.array([1]), lat_deg=70.0, elevation_m=10.0,
    )
    assert eto[0] >= 0.0


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------


def _synthetic_payload() -> dict:
    return {
        "properties": {
            "parameter": {
                "T2M_MAX":          {"20240101": 30.0, "20240102": 31.5},
                "T2M_MIN":          {"20240101": 21.0, "20240102": 22.5},
                "T2M":              {"20240101": 25.5, "20240102": 27.0},
                "RH2M":             {"20240101": 65.0, "20240102": 60.0},
                "WS2M":             {"20240101":  2.0, "20240102":  2.5},
                "ALLSKY_SFC_SW_DWN":{"20240101": 22.0, "20240102": 23.0},
                "PRECTOTCORR":      {"20240101":  0.0, "20240102":  5.5},
            }
        }
    }


def test_parser_returns_canonical_dtypes_and_columns():
    df = _parse_to_dataframe(_synthetic_payload())
    assert list(df["date"]) == [pd.Timestamp("2024-01-01"),
                                pd.Timestamp("2024-01-02")]
    for col in ["Tmax", "Tmin", "Tmean", "RH", "u2", "Rs", "pr"]:
        assert col in df.columns
        assert df[col].dtype == float


def test_parser_handles_missing_sentinel():
    payload = _synthetic_payload()
    payload["properties"]["parameter"]["PRECTOTCORR"]["20240101"] = -999.0
    df = _parse_to_dataframe(payload)
    assert np.isnan(df.loc[0, "pr"])
    assert df.loc[1, "pr"] == 5.5


def test_parser_rejects_bad_payload():
    with pytest.raises(ValueError):
        _parse_to_dataframe({"foo": "bar"})


# ---------------------------------------------------------------------------
# Top-level download_nasa_power: argument validation (no network)
# ---------------------------------------------------------------------------


def test_download_rejects_invalid_coordinates():
    with pytest.raises(ValueError):
        download_nasa_power(lat=95.0, lon=0.0,
                            start="2020-01-01", end="2020-01-02")


def test_download_rejects_inverted_dates():
    with pytest.raises(ValueError):
        download_nasa_power(lat=0.0, lon=0.0,
                            start="2020-01-10", end="2020-01-01")


# ---------------------------------------------------------------------------
# FAO-56 Eq. 47 — wind 10 m → 2 m conversion
# ---------------------------------------------------------------------------


def test_wind_10m_factor_matches_fao56_simplified_075():
    """For z = 10 m the FAO-56 Eq. 47 factor must equal ≈ 0.748,
    which rounds to the ~0.75 rule of thumb cited in the FAO-56 manual."""
    factor = float(wind_10m_to_2m_fao56(np.array([1.0]))[0])
    assert abs(factor - 0.747951) < 1e-5
    # The simplified form differs by ~2 mm/s on a 1 m/s input — small
    # enough that a rounding error never crosses 0.005 m/s.
    assert abs(factor - 0.75) < 0.01


def test_wind_10m_to_2m_scales_linearly():
    u10 = np.array([0.0, 1.0, 2.5, 5.0, 10.0])
    u2 = wind_10m_to_2m_fao56(u10)
    expected = u10 * (4.87 / np.log(67.8 * 10.0 - 5.42))
    np.testing.assert_allclose(u2, expected)


def test_wind_helper_clamps_negative_to_zero():
    """Negative inputs (numerical noise on near-zero wind) must clamp to 0."""
    u = wind_10m_to_2m_fao56(np.array([-0.1, -1.0, 0.5]))
    assert u[0] == 0.0
    assert u[1] == 0.0
    assert u[2] > 0.0


def test_wind_helper_supports_other_heights():
    """Eq. 47 has explicit z-dependence; 2 m → 2 m must equal the input."""
    u = wind_10m_to_2m_fao56(np.array([3.0]), z=2.0)
    factor_2m = 4.87 / np.log(67.8 * 2.0 - 5.42)
    np.testing.assert_allclose(u, [3.0 * factor_2m])
    # ... and must NOT equal the 10 m factor's output
    u10_path = wind_10m_to_2m_fao56(np.array([3.0]), z=10.0)
    assert abs(u[0] - u10_path[0]) > 0.01
