"""Path resolution utilities used throughout the bwb package.

The :func:`find_project_root` helper walks up from the calling file's
location until it finds a marker that identifies the project root. This
keeps scripts/modules location-independent: they work from any working
directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


# Markers that identify the project root, in order of preference.
_DEFAULT_MARKERS = (
    ("data", "extracted_csv"),     # our data folder structure
    ("pyproject.toml",),
    (".git",),
)


def find_project_root(
    start: Path | str | None = None,
    markers: Iterable[Iterable[str]] = _DEFAULT_MARKERS,
) -> Path:
    """Locate the project root by walking up from `start`.

    Parameters
    ----------
    start : Path or str, optional
        Starting directory or file. Defaults to the calling file's directory.
    markers : iterable of iterables of str
        Each inner iterable describes a path (relative to a candidate parent)
        whose existence indicates project root. The first match wins.

    Returns
    -------
    Path
        Absolute path to project root, or the start directory if no marker
        is found (graceful fallback).
    """
    if start is None:
        # Best-effort: start from cwd; callers should pass __file__ for
        # location-independent resolution.
        here = Path.cwd().resolve()
    else:
        start = Path(start).resolve()
        here = start.parent if start.is_file() else start

    for parent in [here] + list(here.parents):
        for marker in markers:
            candidate = parent.joinpath(*marker)
            if candidate.exists():
                return parent
    return here


def project_data_dir(*subpath: str) -> Path:
    """Return `<project_root>/data/<subpath>`."""
    return find_project_root() / "data" / Path(*subpath)


def project_processed_dir(*subpath: str) -> Path:
    """Return `<project_root>/data_processed/<subpath>`."""
    return find_project_root() / "data_processed" / Path(*subpath)


def project_raw_dir(*subpath: str) -> Path:
    """Return `<project_root>/data_raw/<subpath>`."""
    return find_project_root() / "data_raw" / Path(*subpath)


def project_figures_dir(*subpath: str) -> Path:
    """Return `<project_root>/figures/<subpath>`."""
    return find_project_root() / "figures" / Path(*subpath)
