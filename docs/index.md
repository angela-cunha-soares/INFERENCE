# bwb -- Bayesian Water Balance

Documentation for the **bwb** (Bayesian Water Balance) framework: a
hierarchical Bayesian model for stochastic FAO-56 water-balance and
risk-aware irrigation scheduling, validated across the MATOPIBA
agricultural frontier.

## Why a Bayesian framework?

A deterministic FAO-56 baseline answers "what is the soil water content
today?". A Bayesian framework answers "what is the probability that the
soil water content will fall below the critical threshold within the
next five days?" -- which is the question the irrigation operator
actually needs to answer. The full posterior distribution is what makes
risk-aware decisions possible.

## Model in one paragraph

For each crop cycle and city we treat the daily soil-moisture trajectory
as a latent state-space process. Priors on the four hydraulic / crop
parameters (theta_s, theta_r, Kc-multiplier, theta_init) come from
WMO climatological normals and a city x variable x month distribution
atlas. The state evolves daily through the FAO-56 water-balance
recursion; the observation likelihood ties the latent state to a
deterministic FAO-56 reference series with HalfNormal noise. Inference
uses NUTS (PyMC v5 / PyTensor); 5-day forecasts propagate a 31-member
GEFS-like meteorological ensemble through the posterior parameter
draws via nested Monte Carlo.

## Documentation map

* **Quick start**: see the project [README](../README.md).
* **Limitations**: [`limitations.md`](limitations.md).
* **Reproducible notebooks** (paper figures): `notebooks/paper/`.
* **API**: every public symbol is documented via inline docstrings
  (`pydoc bwb.<module>` or look in `src/bwb/`).

## Calibration snapshot

After the 2026-05 calibration update (wider sigma_obs and relaxed
priors on theta_s / theta_init driven by Sobol' sensitivity):

| metric            | value (Balsas / 60 cm / 2023) |
| ----------------- | ----------------------------- |
| RMSE              | 0.008 |
| KGE               | 0.880 |
| NSE               | 0.972 |
| CRPS              | 0.004 |
| coverage_90       | 0.900 |
| pit_alpha         | 0.915 |
| interval_score_90 | 0.033 |

The 90% credible interval is now well calibrated (coverage matches
nominal).

## Cite

Please cite the framework via the metadata in
[`CITATION.cff`](../CITATION.cff).
