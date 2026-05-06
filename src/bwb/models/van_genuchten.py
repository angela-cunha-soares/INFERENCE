"""Van Genuchten soil water retention curve model.

Implements the Van Genuchten (1980) parametric soil hydraulic model
for computing soil water retention and conductivity functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SoilParameters:
    """Van Genuchten soil hydraulic parameters."""
    theta_s: float  # Saturated water content (volumetric)
    theta_r: float  # Residual water content (volumetric)
    alpha: float    # Air entry suction (1/cm)
    n: float        # Pore size distribution index
    m: float        # = 1 - 1/n (Van Genuchten parameter)
    K_s: float      # Saturated hydraulic conductivity (cm/day)


# Default soil parameters for MATOPIBA region
DEFAULT_SOILS = {
    "sandy_loam": SoilParameters(
        theta_s=0.43,
        theta_r=0.045,
        alpha=0.027,
        n=1.41,
        m=0.29,
        K_s=25.0,
    ),
    "loam": SoilParameters(
        theta_s=0.45,
        theta_r=0.08,
        alpha=0.036,
        n=1.56,
        m=0.35,
        K_s=10.0,
    ),
    "clay_loam": SoilParameters(
        theta_s=0.48,
        theta_r=0.10,
        alpha=0.019,
        n=1.31,
        m=0.24,
        K_s=5.0,
    ),
    "sandy_clay_loam": SoilParameters(
        theta_s=0.40,
        theta_r=0.05,
        alpha=0.025,
        n=1.38,
        m=0.28,
        K_s=15.0,
    ),
}


def get_soil_parameters(
    soil_type: str,
    custom_params: Optional[dict] = None,
) -> SoilParameters:
    """Get soil parameters for a given soil type.

    Parameters
    ----------
    soil_type : str
        Soil type name
    custom_params : dict, optional
        Custom parameters to override defaults

    Returns
    -------
    SoilParameters
        Soil hydraulic parameters
    """
    if soil_type in DEFAULT_SOILS:
        params = DEFAULT_SOILS[soil_type]
    else:
        # Return sandy loam as default
        params = DEFAULT_SOILS["sandy_loam"]

    if custom_params:
        return SoilParameters(
            theta_s=custom_params.get("theta_s", params.theta_s),
            theta_r=custom_params.get("theta_r", params.theta_r),
            alpha=custom_params.get("alpha", params.alpha),
            n=custom_params.get("n", params.n),
            m=custom_params.get("m", params.m),
            K_s=custom_params.get("K_s", params.K_s),
        )

    return params


def compute_m(
    n: float,
) -> float:
    """Compute m parameter from n.

    Parameters
    ----------
    n : float
        Pore size distribution index

    Returns
    -------
    float
        m = 1 - 1/n
    """
    return 1 - 1 / n


def water_retention(
    h: np.ndarray,
    params: SoilParameters,
) -> np.ndarray:
    """Compute volumetric water content from matric potential.

    Implements the Van Genuchten (1980) retention equation:
        theta(h) = theta_r + (theta_s - theta_r) / [1 + (alpha * h)^n]^m

    Parameters
    ----------
    h : np.ndarray
        Matric potential (cm) - use positive values
    params : SoilParameters
        Soil hydraulic parameters

    Returns
    -------
    np.ndarray
        Volumetric water content (cm³/cm³)
    """
    m = params.m if params.m else compute_m(params.n)

    # Avoid division by zero
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 1e-10)

    alpha_h = params.alpha * h
    Se = 1 / (1 + alpha_h ** params.n) ** m

    theta = params.theta_r + (params.theta_s - params.theta_r) * Se
    return theta


def inverse_retention(
    theta: np.ndarray,
    params: SoilParameters,
) -> np.ndarray:
    """Compute matric potential from water content.

    Inverts the Van Genuchten equation:
        h = (Se^(-1/m) - 1)^(1/n) / alpha

    Parameters
    ----------
    theta : np.ndarray
        Volumetric water content (cm³/cm³)
    params : SoilParameters
        Soil hydraulic parameters

    Returns
    -------
    np.ndarray
        Matric potential (cm)
    """
    m = params.m if params.m else compute_m(params.n)

    theta = np.asarray(theta, dtype=float)

    # Effective saturation
    Se = (theta - params.theta_r) / (params.theta_s - params.theta_r)
    Se = np.clip(Se, 1e-10, 1.0)

    # Inverse
    h = (Se ** (-1 / m) - 1) ** (1 / params.n) / params.alpha

    return h


def compute_soil_moisture_fraction(
    theta: float,
    params: SoilParameters,
) -> float:
    """Compute soil moisture as fraction of available water.

    Parameters
    ----------
    theta : float
        Volumetric water content
    params : SoilParameters
        Soil parameters

    Returns
    -------
    float
        Fraction of available water (0-1)
    """
    awc = params.theta_s - params.theta_r
    if awc <= 0:
        return 0.0

    fraction = (theta - params.theta_r) / awc
    return float(np.clip(fraction, 0, 1))


def compute_available_water_capacity(
    params: SoilParameters,
    root_depth_cm: float,
) -> float:
    """Compute available water capacity in mm.

    Parameters
    ----------
    params : SoilParameters
        Soil parameters
    root_depth_cm : float
        Root depth in cm

    Returns
    -------
    float
        Available water capacity (mm)
    """
    awc = params.theta_s - params.theta_r
    # Convert cm to mm: depth(cm) * 10 = depth(mm)
    # AWC (cm³/cm³) * depth(mm) = AWC (mm)
    return awc * root_depth_cm * 10


def compute_field_capacity(
    params: SoilParameters,
    pF: float = 2.0,
) -> float:
    """Compute water content at field capacity.

    Parameters
    ----------
    params : SoilParameters
        Soil parameters
    pF : float
        pF value for field capacity (default: 2.0 = -10 kPa)

    Returns
    -------
    float
        Volumetric water content at field capacity
    """
    # Convert pF to matric potential (cm)
    h_fc = 10 ** pF
    return float(water_retention(np.array([h_fc]), params)[0])


def compute_wilting_point(
    params: SoilParameters,
    pF: float = 4.2,
) -> float:
    """Compute water content at permanent wilting point.

    Parameters
    ----------
    params : SoilParameters
        Soil parameters
    pF : float
        pF value for wilting point (default: 4.2 = -1500 kPa)

    Returns
    -------
    float
        Volumetric water content at wilting point
    """
    # Convert pF to matric potential (cm)
    h_wp = 10 ** pF
    return float(water_retention(np.array([h_wp]), params)[0])


def generate_retention_curve(
    params: SoilParameters,
    h_min: float = 1.0,
    h_max: float = 10000.0,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate complete retention curve data.

    Parameters
    ----------
    params : SoilParameters
        Soil parameters
    h_min : float
        Minimum matric potential (cm)
    h_max : float
        Maximum matric potential (cm)
    n_points : int
        Number of points

    Returns
    -------
    tuple
        (h, theta) arrays
    """
    h = np.logspace(np.log10(h_min), np.log10(h_max), n_points)
    theta = water_retention(h, params)
    return h, theta