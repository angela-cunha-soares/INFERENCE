"""Shared pytest fixtures for the bwb test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Ensure src/ is on the import path even when the package is not installed
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260504)


@pytest.fixture(scope="session")
def synthetic_cycle(rng):
    """Synthetic 90-day climate cycle (DataFrame compatible with adapters)."""
    import pandas as pd

    n_days = 90
    p = rng.gamma(1.5, 8.0, n_days) * (rng.uniform(size=n_days) > 0.5)
    eto = np.clip(rng.normal(4.5, 0.8, n_days), 2.0, None)
    dates = pd.date_range("2023-12-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "ETo": eto,
        "pr": p,
        "dia_ciclo": np.arange(1, n_days + 1),
    })


@pytest.fixture(scope="session")
def matopiba_profile():
    from bwb.config.profiles import load_profile
    return load_profile("matopiba")
