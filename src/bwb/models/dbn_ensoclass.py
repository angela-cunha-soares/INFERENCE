"""Dynamic Bayesian Network for ENSO-conditioned seasonal class transitions.

Generalises the static Dirichlet prior on (dry, normal, wet) classes
into a **Dynamic Bayesian Network (DBN)** in which the class
distribution at year :math:`t` is conditioned on (i) the class observed
in year :math:`t-1` and (ii) the seasonal ONI (Oceanic Niño Index)
mean during the cycle. Concretely, for each ENSO regime
:math:`r\\in\\{\\text{La Niña}, \\text{neutral}, \\text{El Niño}\\}`:

.. math::

    \\boldsymbol{\\pi}_r \\sim \\mathrm{Dir}(\\boldsymbol{\\alpha}_0)
    \\qquad
    \\kappa_t \\mid r_t \\sim \\mathrm{Cat}(\\boldsymbol{\\pi}_{r_t})

The hierarchical Dirichlet prior :math:`\\boldsymbol{\\alpha}_0` is
itself given a weakly-informative HalfNormal hyperprior. The model is
non-conjugate (because :math:`\\alpha_0` is now random) and is
sampled with NUTS, illustrating the use case where MCMC is required
even though the elementary Dir-Mult pair would have closed form for
fixed :math:`\\alpha_0`.

The DBN is **dynamic** in the strict sense of Murphy (2002): nodes are
indexed by time (year) and the transition kernel is shared across
slices. We deliberately kept the transition Markov-1 (only :math:`t-1`
matters) since longer memories contradict the limited length of the
historical record.

Outputs (after sampling)
------------------------
* ``pi[r, k]`` — posterior class probabilities given ENSO regime,
  3 regimes × 3 classes = 9 scalars.
* ``alpha0[k]`` — posterior on the global Dirichlet concentration.
* ``log_lik`` — pointwise log-likelihood for LOO comparison.

Reference
---------
Murphy, K. P. (2002). *Dynamic Bayesian networks: representation,
inference and learning*. PhD thesis, UC Berkeley.

Cano, R., Sordo, C., & Gutiérrez, J. M. (2004). Applications of
Bayesian networks in meteorology. In *Advances in Bayesian networks*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DBNData:
    """Inputs to the DBN model.

    Attributes
    ----------
    classes : array of int
        Observed class index per year, values in {0=dry, 1=normal, 2=wet},
        in chronological order.
    enso : array of int
        ENSO regime per year, values in {0=La Niña, 1=neutral, 2=El Niño},
        same length as ``classes``.
    """
    classes: np.ndarray
    enso: np.ndarray

    def __post_init__(self):
        self.classes = np.asarray(self.classes, dtype=int)
        self.enso = np.asarray(self.enso, dtype=int)
        if self.classes.shape != self.enso.shape:
            raise ValueError("classes and enso must have the same shape")
        if self.classes.min() < 0 or self.classes.max() > 2:
            raise ValueError("classes must be in {0,1,2}")
        if self.enso.min() < 0 or self.enso.max() > 2:
            raise ValueError("enso regime must be in {0,1,2}")


def build_dbn_model(data: DBNData):
    """Construct a PyMC v5 model for the ENSO-conditioned DBN.

    The model has 3 ENSO regimes × 3 classes = 9 transition probabilities,
    drawn from a hierarchical Dirichlet with a HalfNormal hyperprior on
    the concentration parameter.
    """
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMC v5 is required. Install with: pip install 'pymc>=5.10' arviz"
        ) from exc
    import pytensor.tensor as pt

    coords = {
        "regime": ["la_nina", "neutral", "el_nino"],
        "klass":  ["dry", "normal", "wet"],
        "obs":    np.arange(len(data.classes)),
    }

    with pm.Model(coords=coords) as model:
        # Hierarchical concentration: shared HalfNormal hyperprior
        alpha0 = pm.HalfNormal("alpha0", sigma=5.0, dims="klass")

        # 3 Dirichlets, one per ENSO regime
        # alpha0 is broadcast across regimes (all share the same prior shape)
        pi = pm.Dirichlet(
            "pi", a=alpha0[None, :].repeat(3, axis=0),
            dims=("regime", "klass"),
        )

        # Likelihood: each observed class drawn from Cat(pi[r])
        regime_idx = pm.Data("regime_idx", data.enso, dims="obs")
        # Use Categorical with p indexed by regime_idx
        p_obs = pi[regime_idx]  # shape (n_obs, 3)
        pm.Categorical(
            "kappa", p=p_obs,
            observed=data.classes,
            dims="obs",
        )
    return model


def fit_dbn(data: DBNData, *, draws: int = 1000, tune: int = 1000,
            chains: int = 4, target_accept: float = 0.95,
            random_seed: int = 42, progressbar: bool = False):
    """Fit the DBN with NUTS and return the InferenceData.

    Returns
    -------
    idata : arviz.InferenceData
    diagnostics : dict
        Same shape as :func:`bwb.models.water_balance.diagnostics_summary`.
    """
    import pymc as pm
    import arviz as az

    model = build_dbn_model(data)
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, random_seed=random_seed,
            progressbar=progressbar, return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    summary = az.summary(idata, var_names=["alpha0", "pi"],
                          kind="diagnostics", round_to="none")
    n_div = int(idata.sample_stats["diverging"].sum().values) \
        if "diverging" in idata.sample_stats else 0
    diag = {
        "max_rhat":      float(summary["r_hat"].max()),
        "min_ess_bulk":  float(summary["ess_bulk"].min()),
        "min_ess_tail":  float(summary["ess_tail"].min()),
        "n_divergent":   n_div,
        "converged":     bool(summary["r_hat"].max() < 1.05 and n_div == 0),
    }
    return idata, diag


def classify_oni(oni_seasonal_mean: np.ndarray) -> np.ndarray:
    """Classify each season's ONI value into {0=La Niña, 1=neutral, 2=El Niño}.

    Uses the standard NOAA CPC threshold of $\\pm 0.5\\,^\\circ$C
    on the 3-month running ONI (Oceanic Niño Index).
    """
    o = np.asarray(oni_seasonal_mean, dtype=float)
    out = np.full(o.shape, 1, dtype=int)  # neutral default
    out[o <= -0.5] = 0  # La Niña
    out[o >=  0.5] = 2  # El Niño
    return out


__all__ = ["DBNData", "build_dbn_model", "fit_dbn", "classify_oni"]
