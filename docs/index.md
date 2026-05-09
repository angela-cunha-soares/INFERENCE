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
parameters (theta_s, theta_r, Kc-multiplier, theta_init) come from the
SNIRH-CAD national soil dataset and the FAO-56 crop library. The state
evolves daily through the FAO-56 water-balance recursion; the
observation likelihood ties the latent state to a deterministic FAO-56
reference series with HalfNormal noise. Inference uses NUTS (PyMC v5
/ PyTensor) for the non-conjugate hydraulic state-space and a closed-form
conjugate Dirichlet--Multinomial update for the SPEI-tercile yearly
class weights, the latter initialised with a climatologically-informed
prior on the 1961-2019 training window. Operational rolling 5-day
forecasts propagate the SPEI-weighted analogue ensemble through the
same FAO-56 recursion.

## Documentation map

* **Quick start**: see the project [README](../README.md).
* **Methodology**: [`methodology.md`](methodology.md) (three validation
  modes, DAG learning, ENSO-conditioned DBN, fused product) and
  [`methodology_diagram.md`](methodology_diagram.md) (visual pipeline).
* **Glossary**: [`acronyms_and_methods.md`](acronyms_and_methods.md).
* **Limitations**: [`limitations.md`](limitations.md) (six explicit
  limitations + calibration caveats).
* **Reproducible notebooks** (paper figures): `notebooks/paper/`.
* **API**: every public symbol is documented via inline docstrings
  (`pydoc bwb.<module>` or look in `src/bwb/`).

## Calibration snapshot (2026-05, climatologically-informed prior)

The framework reports calibrated probabilistic skill across three
complementary validation modes:

### A. Posterior-recovery test (150 combinations: 10 cities × 3 depths × 5 cycles)

| Metric            | mean  | median | min   | max   |
| ----------------- | ----- | ------ | ----- | ----- |
| KGE               | 0.909 | 0.918  | 0.798 | 0.985 |
| NSE               | 0.976 | 0.979  | 0.933 | 0.997 |
| CRPS (m³/m³)      | 0.005 | 0.005  | 0.002 | 0.010 |
| α-PIT             | 0.894 | 0.902  | 0.749 | 0.972 |
| coverage_90       | 0.918 | 0.922  | 0.856 | 0.978 |

### B. Climatological sequential forecast (50 combinations: 10 cities × 5 cycles)

| Metric                                | mean  | median |
| ------------------------------------- | ----- | ------ |
| CRPS on seasonal irrigation (mm)      | 20.63 | 18.18  |
| coverage_90 daily soil-water content  | 0.954 | 0.956  |
| coverage_90 seasonal irrigation depth | 0.96  | 1.00   |
| KGE on daily soil-water content       | -0.10 | -0.09  |
| PIT on seasonal irrigation depth      | 0.46  | 0.41   |
| CRPSS vs naive climatological mean    | +0.03 | +0.04  |

### C. Operational rolling 5-day forecast (4,250 forecasts)

| Metric                                            | aggregate |
| ------------------------------------------------- | --------- |
| Median KGE on 5-day cumulative irrigation         | +0.32     |
| Mean CRPS (mm)                                    | 2.65      |
| Empirical 90% prediction-interval coverage        | 0.970     |

The 90% credible / prediction intervals are well-calibrated to
conservatively over-covered on all three modes; the climatological-mode
KGE is negative by design (a climatological forecast does not target
the realised daily sequence). For risk-aware irrigation in a
high-interannual-variability region like MATOPIBA, **calibration of
the credible intervals is the operational metric that matters most**;
see the manuscript Discussion (Section "Operational interpretation:
calibration first, raw skill second") for the rationale.

## Cite

Please cite the framework via the metadata in
[`CITATION.cff`](../CITATION.cff).
