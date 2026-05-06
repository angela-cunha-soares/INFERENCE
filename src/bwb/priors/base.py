"""Base classes for prior distributions in the Bayesian water balance framework.

Provides abstract base class and common functionality for all prior
distribution implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PriorConfig:
    """Configuration for prior distributions."""
    name: str
    family: str
    params: dict
    bounds: tuple[float, float]


class BasePrior(ABC):
    """Abstract base class for prior distributions.

    All prior implementations should inherit from this class and
    implement the required abstract methods.
    """

    def __init__(
        self,
        family: str,
        params: Optional[dict] = None,
        bounds: Optional[tuple[float, float]] = None,
    ):
        """Initialize prior.

        Parameters
        ----------
        family : str
            Distribution family name
        params : dict, optional
            Distribution parameters
        bounds : tuple[float, float], optional
            Valid bounds for the distribution
        """
        self.family = family
        self.params = params or {}
        self.bounds = bounds or (-np.inf, np.inf)

    @abstractmethod
    def sample(self, n: int, random_seed: Optional[int] = None) -> np.ndarray:
        """Generate samples from the prior.

        Parameters
        ----------
        n : int
            Number of samples
        random_seed : int, optional
            Random seed

        Returns
        -------
        np.ndarray
            Samples from the prior
        """
        pass

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density.

        Parameters
        ----------
        x : np.ndarray
            Values at which to evaluate the PDF

        Returns
        -------
        np.ndarray
            PDF values
        """
        pass

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution function.

        Parameters
        ----------
        x : np.ndarray
            Values at which to evaluate the CDF

        Returns
        -------
        np.ndarray
            CDF values
        """
        pass

    def mean(self) -> float:
        """Compute mean of the prior."""
        raise NotImplementedError

    def std(self) -> float:
        """Compute standard deviation of the prior."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Convert prior to dictionary representation."""
        return {
            "family": self.family,
            "params": self.params,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BasePrior":
        """Create prior from dictionary representation."""
        raise NotImplementedError


class NormalPrior(BasePrior):
    """Normal (Gaussian) prior distribution."""

    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        bounds: Optional[tuple[float, float]] = None,
    ):
        super().__init__("norm", {"loc": loc, "scale": scale}, bounds)
        self.loc = loc
        self.scale = scale

    def sample(self, n: int, random_seed: Optional[int] = None) -> np.ndarray:
        if random_seed is not None:
            np.random.seed(random_seed)
        return np.random.normal(self.loc, self.scale, n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.norm.pdf(x, self.loc, self.scale)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.norm.cdf(x, self.loc, self.scale)

    def mean(self) -> float:
        return self.loc

    def std(self) -> float:
        return self.scale


class TruncatedNormalPrior(BasePrior):
    """Truncated normal prior with specified bounds."""

    def __init__(
        self,
        loc: float,
        scale: float,
        lower: float,
        upper: float,
    ):
        super().__init__("truncnorm", {"loc": loc, "scale": scale}, (lower, upper))
        self.loc = loc
        self.scale = scale
        self.lower = lower
        self.upper = upper

    def sample(self, n: int, random_seed: Optional[int] = None) -> np.ndarray:
        from scipy import stats
        if random_seed is not None:
            np.random.seed(random_seed)
        a = (self.lower - self.loc) / self.scale
        b = (self.upper - self.loc) / self.scale
        return stats.truncnorm.rvs(a, b, loc=self.loc, scale=self.scale, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        a = (self.lower - self.loc) / self.scale
        b = (self.upper - self.loc) / self.scale
        return stats.truncnorm.pdf(x, a, b, loc=self.loc, scale=self.scale)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        a = (self.lower - self.loc) / self.scale
        b = (self.upper - self.loc) / self.scale
        return stats.truncnorm.cdf(x, a, b, loc=self.loc, scale=self.scale)

    def mean(self) -> float:
        from scipy import stats
        a = (self.lower - self.loc) / self.scale
        b = (self.upper - self.loc) / self.scale
        return stats.truncnorm.mean(a, b, loc=self.loc, scale=self.scale)


class GammaPrior(BasePrior):
    """Gamma prior distribution."""

    def __init__(
        self,
        shape: float,
        scale: float,
        bounds: Optional[tuple[float, float]] = (0, np.inf),
    ):
        super().__init__("gamma", {"shape": shape, "scale": scale}, bounds)
        self.shape = shape
        self.scale = scale

    def sample(self, n: int, random_seed: Optional[int] = None) -> np.ndarray:
        if random_seed is not None:
            np.random.seed(random_seed)
        return np.random.gamma(self.shape, self.scale, n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.gamma.pdf(x, self.shape, scale=self.scale)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.gamma.cdf(x, self.shape, scale=self.scale)

    def mean(self) -> float:
        return self.shape * self.scale

    def std(self) -> float:
        return np.sqrt(self.shape) * self.scale


class BetaPrior(BasePrior):
    """Beta prior distribution."""

    def __init__(
        self,
        a: float,
        b: float,
        bounds: tuple[float, float] = (0, 1),
    ):
        super().__init__("beta", {"a": a, "b": b}, bounds)
        self.a = a
        self.b = b

    def sample(self, n: int, random_seed: Optional[int] = None) -> np.ndarray:
        if random_seed is not None:
            np.random.seed(random_seed)
        return np.random.beta(self.a, self.b, n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.beta.pdf(x, self.a, self.b)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        from scipy import stats
        return stats.beta.cdf(x, self.a, self.b)

    def mean(self) -> float:
        return self.a / (self.a + self.b)


def create_prior(family: str, params: dict) -> BasePrior:
    """Factory function to create prior instances.

    Parameters
    ----------
    family : str
        Distribution family
    params : dict
        Distribution parameters

    Returns
    -------
    BasePrior
        Prior instance
    """
    if family == "norm":
        return NormalPrior(params.get("loc", 0), params.get("scale", 1))
    elif family == "truncnorm":
        return TruncatedNormalPrior(
            params["loc"],
            params["scale"],
            params["lower"],
            params["upper"],
        )
    elif family == "gamma":
        return GammaPrior(params["shape"], params["scale"])
    elif family == "beta":
        return BetaPrior(params["a"], params["b"])
    else:
        raise ValueError(f"Unknown prior family: {family}")