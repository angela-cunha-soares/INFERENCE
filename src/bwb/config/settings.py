"""Global settings for the bwb framework.

We use plain dataclasses (no Pydantic dependency) so the package stays light
and importable even in minimal environments. The :func:`get_settings` helper
returns a singleton populated from environment variables, which keeps the
behaviour configurable in CI/Docker without a separate config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_path(name: str, default: Optional[Path]) -> Optional[Path]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return Path(raw).expanduser().resolve()


@dataclass
class GlobalSettings:
    """Runtime configuration for the bwb framework."""

    # Paths (resolved lazily via bwb.utils.paths if not set)
    project_root: Optional[Path] = None
    data_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    raw_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    figures_dir: Optional[Path] = None

    # Bayesian sampling defaults
    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    random_seed: int = 42

    # Forecast / sensitivity
    n_simulations: int = 500
    n_sensitivity_samples: int = 1000

    # Active region (matches a TOML profile name in config/regional/)
    region: str = "matopiba"

    # Misc
    log_level: str = "INFO"
    progressbar: bool = True

    # Free-form extras populated from a profile (cities, kc params, etc.)
    profile: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("project_root", "data_dir", "processed_dir", "raw_dir",
                    "output_dir", "figures_dir"):
            v = d.get(key)
            if isinstance(v, Path):
                d[key] = str(v)
        return d


def _build_default() -> GlobalSettings:
    from bwb.utils.paths import find_project_root

    root = _env_path("BWB_PROJECT_ROOT", None)
    if root is None:
        root = find_project_root()

    return GlobalSettings(
        project_root=root,
        data_dir=_env_path("BWB_DATA_DIR", root / "data"),
        processed_dir=_env_path("BWB_PROCESSED_DIR", root / "data_processed"),
        raw_dir=_env_path("BWB_RAW_DIR", root / "data_raw"),
        output_dir=_env_path("BWB_OUTPUT_DIR", root / "output"),
        figures_dir=_env_path("BWB_FIGURES_DIR", root / "figures"),
        draws=_env_int("BWB_DRAWS", 2000),
        tune=_env_int("BWB_TUNE", 1000),
        chains=_env_int("BWB_CHAINS", 4),
        target_accept=_env_float("BWB_TARGET_ACCEPT", 0.95),
        random_seed=_env_int("BWB_SEED", 42),
        n_simulations=_env_int("BWB_N_SIMULATIONS", 500),
        n_sensitivity_samples=_env_int("BWB_N_SENS_SAMPLES", 1000),
        region=os.environ.get("BWB_REGION", "matopiba"),
        log_level=os.environ.get("BWB_LOG_LEVEL", "INFO"),
        progressbar=os.environ.get("BWB_PROGRESSBAR", "1") not in ("0", "false", "False"),
    )


_SETTINGS: Optional[GlobalSettings] = None


def get_settings(reload: bool = False) -> GlobalSettings:
    """Return the singleton :class:`GlobalSettings` (lazy)."""
    global _SETTINGS
    if _SETTINGS is None or reload:
        _SETTINGS = _build_default()
    return _SETTINGS


def reset_settings() -> None:
    """Drop the cached singleton (useful for tests)."""
    global _SETTINGS
    _SETTINGS = None
