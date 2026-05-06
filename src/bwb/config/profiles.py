"""Regional profile loader.

Profiles live in :mod:`bwb.config.regional` as TOML files. The active region is
selected via :class:`bwb.config.settings.GlobalSettings.region` (default
``"matopiba"``).

Usage
-----
>>> from bwb.config.profiles import load_profile
>>> profile = load_profile("matopiba")
>>> profile["crop"]["awc_mm"]
120
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]


_REGIONAL_DIR = Path(__file__).resolve().parent / "regional"


class ProfileNotFoundError(FileNotFoundError):
    """Raised when a regional profile cannot be located."""


def list_profiles() -> list[str]:
    """Return the names of all bundled regional profiles."""
    if not _REGIONAL_DIR.exists():
        return []
    return sorted(p.stem for p in _REGIONAL_DIR.glob("*.toml"))


def profile_path(name: str) -> Path:
    """Resolve the path to a profile TOML file."""
    p = _REGIONAL_DIR / f"{name}.toml"
    if not p.exists():
        raise ProfileNotFoundError(
            f"No profile named {name!r} in {_REGIONAL_DIR}. "
            f"Available: {list_profiles()}"
        )
    return p


def load_profile(name: str = "matopiba", path: Optional[Path] = None) -> dict[str, Any]:
    """Load a regional profile from TOML.

    Parameters
    ----------
    name : str
        Profile name (matches a TOML file in ``config/regional/``).
    path : Path, optional
        Explicit path to a profile file (overrides ``name``).
    """
    target = Path(path) if path is not None else profile_path(name)
    if not target.exists():
        raise ProfileNotFoundError(f"Profile file not found: {target}")
    with target.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Profile {target} did not parse to a mapping")
    return data


def merge_profile(settings, name: Optional[str] = None) -> None:
    """Load a profile and attach it to ``settings.profile`` in-place."""
    selected = name or getattr(settings, "region", "matopiba")
    settings.profile = load_profile(selected)
