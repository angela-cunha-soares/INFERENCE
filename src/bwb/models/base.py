"""Base classes for water balance models.

Provides abstract base class and common interfaces for all water
balance model implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for water balance models."""
    city: str
    soil_depth_cm: int
    crop_cycle: int
    awc_mm: float
    root_depth_cm: int


@dataclass
class ModelState:
    """State container for water balance model."""
    date: np.ndarray
    precipitation: np.ndarray
    eto: np.ndarray
    soil_moisture: np.ndarray
    irrigation: np.ndarray
    runoffs: np.ndarray


class BaseWaterBalanceModel(ABC):
    """Abstract base class for water balance models.

    All water balance model implementations should inherit from
    this class and implement the required abstract methods.
    """

    def __init__(self, config: ModelConfig):
        """Initialize model.

        Parameters
        ----------
        config : ModelConfig
            Model configuration
        """
        self.config = config
        self.state: Optional[ModelState] = None

    @abstractmethod
    def run(
        self,
        precipitation: np.ndarray,
        eto: np.ndarray,
        initial_soil_moisture: Optional[float] = None,
    ) -> ModelState:
        """Run the water balance model.

        Parameters
        ----------
        precipitation : np.ndarray
            Daily precipitation (mm)
        eto : np.ndarray
            Daily reference evapotranspiration (mm)
        initial_soil_moisture : float, optional
            Initial soil moisture (fraction of AWC)

        Returns
        -------
        ModelState
            Model results
        """
        pass

    @abstractmethod
    def compute_etc(
        self,
        kc: np.ndarray,
        eto: np.ndarray,
    ) -> np.ndarray:
        """Compute crop evapotranspiration.

        Parameters
        ----------
        kc : np.ndarray
            Crop coefficients
        eto : np.ndarray
            Reference evapotranspiration

        Returns
        -------
        np.ndarray
            Crop evapotranspiration
        """
        pass

    @abstractmethod
    def compute_drainage(
        self,
        soil_moisture: np.ndarray,
    ) -> np.ndarray:
        """Compute drainage/ runoff.

        Parameters
        ----------
        soil_moisture : np.ndarray
            Soil moisture states

        Returns
        -------
        np.ndarray
            Drainage amounts
        """
        pass

    def get_state(self) -> Optional[ModelState]:
        """Get current model state."""
        return self.state

    def reset(self):
        """Reset model state."""
        self.state = None


class DeterministicModel(BaseWaterBalanceModel):
    """Deterministic FAO-56 water balance model."""

    def run(
        self,
        precipitation: np.ndarray,
        eto: np.ndarray,
        initial_soil_moisture: Optional[float] = None,
    ) -> ModelState:
        """Run deterministic water balance."""
        n_days = len(precipitation)

        if initial_soil_moisture is None:
            initial_soil_moisture = 0.5  # 50% of AWC

        # Initialize arrays
        soil_moisture = np.zeros(n_days)
        irrigation = np.zeros(n_days)
        runoffs = np.zeros(n_days)

        # Initial soil moisture in mm
        soil_moisture[0] = initial_soil_moisture * self.config.awc_mm

        for d in range(1, n_days):
            # Water balance
            delta = precipitation[d] - eto[d]

            # Update soil moisture
            new_moisture = soil_moisture[d - 1] + delta

            # Clip to available range
            soil_moisture[d] = np.clip(
                new_moisture,
                0,
                self.config.awc_mm,
            )

            # Simple runoff estimation
            if new_moisture > self.config.awc_mm:
                runoffs[d] = new_moisture - self.config.awc_mm
                soil_moisture[d] = self.config.awc_mm

        self.state = ModelState(
            date=np.arange(n_days),
            precipitation=precipitation,
            eto=eto,
            soil_moisture=soil_moisture,
            irrigation=irrigation,
            runoffs=runoffs,
        )

        return self.state

    def compute_etc(
        self,
        kc: np.ndarray,
        eto: np.ndarray,
    ) -> np.ndarray:
        """Compute ETc = Kc × ETo."""
        n = min(len(kc), len(eto))
        return kc[:n] * eto[:n]

    def compute_drainage(
        self,
        soil_moisture: np.ndarray,
    ) -> np.ndarray:
        """Compute drainage when soil exceeds field capacity."""
        drainage = np.maximum(0, soil_moisture - self.config.awc_mm * 0.9)
        return drainage


def create_model(
    model_type: str = "deterministic",
    config: Optional[ModelConfig] = None,
) -> BaseWaterBalanceModel:
    """Factory function to create model instances.

    Parameters
    ----------
    model_type : str
        Model type ('deterministic', 'bayesian')
    config : ModelConfig, optional
        Model configuration

    Returns
    -------
    BaseWaterBalanceModel
        Model instance
    """
    if config is None:
        config = ModelConfig(
            city="Balsas",
            soil_depth_cm=60,
            crop_cycle=2020,
            awc_mm=120,
            root_depth_cm=60,
        )

    if model_type == "deterministic":
        return DeterministicModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")