# Limitations and known issues

Anticipating the limits of a model is what distinguishes a *good* thesis
from an *excellent* one. The four sections below document the boundaries
of what bwb claims and what we explicitly do **not** claim.

## 1. Latency

* **NUTS sampling cost.** Each city x soil-depth x cycle combination
  takes 15--30 s on a recent laptop (2 chains of 300 draws each, after
  300 tune steps). The full 150-combination grid runs in roughly
  35--75 minutes sequentially. This is fine for retrospective
  validation but unsuitable for sub-minute irrigation alerts.
* **Mitigation.** For real-time alerts, fit the model offline and reuse
  the posterior in the ensemble propagator (`bwb.forecast.ensemble`),
  which is sub-second. Variational inference (`bwb.inference.vi`) is
  exposed as a fast-path preview but is *not* used for the published
  results.

## 2. Data dependencies

* **Reference observation series.** The Bayesian likelihood currently
  uses a deterministic FAO-56 baseline as the "observation". This is a
  pragmatic choice driven by data availability -- in-situ continuous
  soil-moisture measurements for the 10 MATOPIBA cities are not openly
  available, and the Xavier reanalysis (Xavier et al. 2022) provides
  meteorological forcings (P, ETo, T, RH, Rs, u2) but **does not include
  soil moisture**. The framework therefore validates the *consistency*
  of the posterior with the FAO-56 reference, not against ground truth.
  Candidate soil-moisture products for an external skill score include
  **GLEAM SMroot** (root-zone, 0.25 deg, 1980-present), **ESA CCI SM**
  (surface only, 1978-present), and **GLDAS-2.x Noah-LSM**. Integrating
  any of these is on the roadmap.
* **Oceanic indices.** Only ONI is currently consolidated. MEI raw is
  shipped but not yet aligned monthly; AMM, AMO, PDO, MJO and IOD are
  documented but not downloaded. The consolidator gracefully handles
  missing indices, so the validation grid runs without them.
* **Soil parameters.** `data/soils/soilgrids.py` is a stub. Three
  textural classes (sandy_loam, loam, clay_loam) are hard-coded in
  `bwb.models.van_genuchten`; results outside this textural envelope
  should be treated cautiously.

## 3. Spatial generalisation

* **Calibrated only on MATOPIBA.** The hierarchical priors and the
  default `matopiba.toml` profile were calibrated on the 10 cities of
  the MATOPIBA agricultural frontier. Applying the framework to other
  regions requires authoring a new TOML profile (cities, planting
  window, soil texture, prior widths) and recomputing the
  climatological normals + distribution atlas (`compute_climatological_
  normals.py` and `identify_distributions.py`).
* **One crop, one cycle length per profile.** The shipped profile uses
  90-day soybean. Other crops / cycle lengths require a new entry in
  the FAO-56 library (`data/crops/fao56_table12_kc.json`) and a
  matching `[crop]` block in the regional TOML.

## 4. Climate non-stationarity

* **Stationary priors.** The climatological priors are derived from
  three WMO normals (1961-1990, 1981-2010, 1991-2020) but the model
  treats each cycle's parameters as exchangeable across the validation
  grid. We do not (yet) parameterise long-term trends or include
  decadal teleconnections (PDO, AMO) as covariates.
* **Mitigation pathway.** `bwb.priors.oceanic` exposes the ENSO ONI
  series and the framework's TOML profile is wired to plug in
  per-cycle covariates. Adding a hierarchical regression on
  decadal-scale predictors is a tractable extension; we leave it for a
  follow-up paper.
* **Ensemble forecast.** `bwb.forecast.ensemble.synthetic_perturbed_ensemble`
  is a placeholder until real GEFS reforecast tiles are integrated. The
  multiplicative log-normal perturbations preserve the climatology of
  the deterministic forcings but do not capture spatially correlated
  storm-scale errors.

## Calibration caveats

* The 90% coverage achieved on the validation grid (~0.90) is *self-
  consistent*: it tests that the Bayesian posterior captures the
  deterministic FAO-56 reference, not that it captures field-scale
  reality. Replace the reference series with measured soil moisture
  before quoting the coverage as an external skill score.
* Sobol' sensitivity attributes ~73% of the seasonal-deficit variance
  to theta_s alone. The widened theta_s prior (sigma=0.04) lets the
  data move the posterior, but the model is still strongly identified
  along the theta_r / Kc_mult axes -- those parameters are weakly
  observable in the soil-moisture trace alone.
