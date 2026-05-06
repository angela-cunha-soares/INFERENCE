"""Crop phenology loader and utilities.

Loads crop coefficient data from the FAO-56 crop library and provides
utilities for managing crop phenological stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CropStage:
    """Single crop phenological stage."""
    name: str
    start_day: int
    end_day: int
    kc: float
    description: str


@dataclass
class Crop:
    """Complete crop phenology definition."""
    name: str
    variety: str
    total_cycle_days: int
    stages: list[CropStage]
    root_depth_cm: int
    awc_mm: float


def load_crop_library(
    library_path: Optional[Path] = None,
) -> dict:
    """Load crop coefficient library.

    Parameters
    ----------
    library_path : Path, optional
        Path to crop library JSON

    Returns
    -------
    dict
        Crop library data
    """
    if library_path is None:
        from bwb.utils.paths import project_data_dir
        library_path = project_data_dir() / "crops" / "fao56_table12_kc.json"
    else:
        library_path = Path(library_path)

    if not library_path.exists():
        # Return default soybean parameters
        return get_default_soybean_library()

    with open(library_path) as f:
        return json.load(f)


def get_default_soybean_library() -> dict:
    """Get default soybean crop parameters (FAO-56 Table 12)."""
    return {
        "soybean": {
            "varieties": {
                "early": {
                    "cycle_days": 90,
                    "kc": {
                        "initial": 0.40,
                        "development": 0.80,
                        "mid": 1.15,
                        "late": 0.80,
                        "harvest": 0.50,
                    },
                    "stage_lengths": {
                        "initial": 15,
                        "development": 15,
                        "mid": 40,
                        "late": 20,
                    },
                    "root_depth_cm": 60,
                    "awc_mm": 120,
                },
                "medium": {
                    "cycle_days": 100,
                    "kc": {
                        "initial": 0.40,
                        "development": 0.80,
                        "mid": 1.15,
                        "late": 0.80,
                        "harvest": 0.50,
                    },
                    "stage_lengths": {
                        "initial": 20,
                        "development": 20,
                        "mid": 40,
                        "late": 20,
                    },
                    "root_depth_cm": 60,
                    "awc_mm": 120,
                },
                "late": {
                    "cycle_days": 120,
                    "kc": {
                        "initial": 0.40,
                        "development": 0.80,
                        "mid": 1.15,
                        "late": 0.80,
                        "harvest": 0.50,
                    },
                    "stage_lengths": {
                        "initial": 25,
                        "development": 25,
                        "mid": 45,
                        "late": 25,
                    },
                    "root_depth_cm": 60,
                    "awc_mm": 120,
                },
            },
        },
    }


def load_crop(
    crop_name: str = "soybeans",
    variety: str = "early",
    library: Optional[dict] = None,
) -> Crop:
    """Load crop definition from library.

    Parameters
    ----------
    crop_name : str
        Crop name (default: 'soybeans')
    variety : str
        Variety name (default: 'early')
    library : dict, optional
        Crop library (loads default if None)

    Returns
    -------
    Crop
        Crop definition
    """
    if library is None:
        library = load_crop_library()

    # The JSON has crops under 'crops' key
    crops_dict = library.get("crops", library)
    
    if crop_name not in crops_dict:
        # Try fallback to default library
        default_lib = get_default_soybean_library()
        if crop_name in default_lib.get("crops", default_lib):
            crops_dict = default_lib.get("crops", default_lib)
            library = default_lib
        else:
            raise ValueError(f"Crop not found: {crop_name}")

    crop_data = crops_dict[crop_name]

    if "varieties" in crop_data:
        if variety not in crop_data["varieties"]:
            raise ValueError(f"Variety not found: {variety}")
        var_data = crop_data["varieties"][variety]
    else:
        var_data = crop_data

    # Build stages
    stage_lengths = var_data.get("stage_lengths", {})
    kc_values = var_data.get("kc", {})

    stages = []
    day = 0
    for stage_name, length in stage_lengths.items():
        kc_key = stage_name if stage_name in kc_values else stage_name[:3]
        kc = kc_values.get(kc_key, 0.4)

        stages.append(CropStage(
            name=stage_name,
            start_day=day,
            end_day=day + length,
            kc=kc,
            description=f"Crop stage: {stage_name}",
        ))
        day += length

    return Crop(
        name=crop_name,
        variety=variety,
        total_cycle_days=var_data.get("cycle_days", 90),
        stages=stages,
        root_depth_cm=var_data.get("root_depth_cm", 60),
        awc_mm=var_data.get("awc_mm", 120),
    )


def get_kc_for_day(crop: Crop, day: int) -> float:
    """Get crop coefficient for a specific day.

    Parameters
    ----------
    crop : Crop
        Crop definition
    day : int
        Day of crop cycle (0-indexed)

    Returns
    -------
    float
        Crop coefficient for that day
    """
    for stage in crop.stages:
        if stage.start_day <= day < stage.end_day:
            return stage.kc
    return crop.stages[-1].kc if crop.stages else 0.4


def get_kc_array(crop: Crop) -> np.ndarray:
    """Get array of Kc values for entire crop cycle.

    Parameters
    ----------
    crop : Crop
        Crop definition

    Returns
    -------
    np.ndarray
        Kc values for each day of the cycle
    """
    return np.array([get_kc_for_day(crop, d) for d in range(crop.total_cycle_days)])


def compute_crop_evapotranspiration(
    crop: Crop,
    eto: np.ndarray,
) -> np.ndarray:
    """Compute crop evapotranspiration (ETc) from reference ETo.

    Parameters
    ----------
    crop : Crop
        Crop definition
    eto : np.ndarray
        Reference evapotranspiration (mm/day)

    Returns
    -------
    np.ndarray
        Crop evapotranspiration (mm/day)
    """
    kc = get_kc_array(crop)
    n_days = min(len(eto), len(kc))
    return eto[:n_days] * kc[:n_days]