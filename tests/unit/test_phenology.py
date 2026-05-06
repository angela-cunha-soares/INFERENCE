"""Unit tests for bwb.phenology."""

from __future__ import annotations

import numpy as np
import pytest

from bwb.phenology.kc_curves import (
    fao56_kc_curve, soybean_kc_90d, kc_daily_from_crop_dict,
)
from bwb.phenology.crop import (
    Crop, get_kc_array, get_kc_for_day, load_crop, get_default_soybean_library,
)


def test_kc_curve_lengths_and_endpoints():
    kc = soybean_kc_90d()
    assert len(kc) == 90
    assert kc[0] == pytest.approx(0.40)        # ini
    assert kc[60] == pytest.approx(1.15)       # mid
    assert kc[-1] == pytest.approx(0.50, abs=1e-6)


def test_kc_curve_is_monotone_within_each_segment():
    kc = soybean_kc_90d()
    assert np.all(np.diff(kc[:15]) == 0)
    assert np.all(np.diff(kc[15:30]) >= 0)
    assert np.allclose(np.diff(kc[30:70]), 0)
    assert np.all(np.diff(kc[70:]) <= 0)


def test_kc_curve_from_dict_with_range_values():
    crop = {"kc": {"ini": {"low": 0.3, "high": 0.5},
                   "mid": 1.15, "end": {"low": 0.4, "high": 0.6}}}
    kc = kc_daily_from_crop_dict(crop, L_ini=15, L_dev=15, L_mid=40, L_late=20)
    assert len(kc) == 90
    assert kc[0] == pytest.approx(0.40)
    assert kc[-1] == pytest.approx(0.50)


def test_kc_curve_zero_lengths():
    assert len(fao56_kc_curve(0.4, 1.15, 0.5, 0, 0, 0, 0)) == 0
    with pytest.raises(ValueError):
        fao56_kc_curve(0.4, 1.15, 0.5, -1, 15, 40, 20)


def test_load_default_soybean_crop():
    crop = load_crop("soybean", "early", library=get_default_soybean_library())
    assert isinstance(crop, Crop)
    assert crop.total_cycle_days == 90
    assert crop.root_depth_cm == 60
    assert len(crop.stages) == 4


def test_get_kc_for_day_within_stage():
    crop = load_crop("soybean", "early", library=get_default_soybean_library())
    kc_arr = get_kc_array(crop)
    assert len(kc_arr) == crop.total_cycle_days
    assert get_kc_for_day(crop, 0) == pytest.approx(0.40)
