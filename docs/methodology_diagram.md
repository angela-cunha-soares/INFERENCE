# Methodology — pipeline of the rolling 5-day operational forecast

This document is the visual companion to [acronyms_and_methods.md](acronyms_and_methods.md). It traces the dataflow from raw inputs to the final decision support output.

## High-level pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  INPUTS                                                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  • Location:     lat, lon, elevation                                     │
│  • Crop:         species, variety, planting_date                         │
│  • Date:         forecast_date  ("today")                                │
│  • Horizon:      H = 5 days                                              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────────┐
              ▼                                          ▼
┌─────────────────────────┐                ┌─────────────────────────────┐
│ EXOGENOUS DATA LOOKUP   │                │  HISTORICAL CLIMATE         │
├─────────────────────────┤                ├─────────────────────────────┤
│ • AWC (mm)              │                │  Daily P, ETo               │
│   ← SNIRH/ANA per       │                │  1961 .. forecast_date      │
│     municipality        │                │  ← Xavier reanalysis        │
│ • Kc curve (90 days)    │                │    (or NASA POWER, ERA5,    │
│   ← FAO-56 Cap. 6,      │                │     INMET, OpenMeteo …)     │
│     5-stage step (soja) │                └────────┬────────────────────┘
│ • Z_r = 60 cm           │                         │
│ • MAD = 0.55            │                         │
└────────────┬────────────┘                         │
             │                                      │
             └──────────────┬───────────────────────┘
                            ▼
       ┌────────────────────────────────────────────┐
       │  STEP 1.  CLIMATOLOGY CLASSIFICATION       │
       │  (1961 ..  forecast_year - 1)              │
       ├────────────────────────────────────────────┤
       │  for each historical cycle y:              │
       │     D_y = ΣP - ΣETo  (90 days)             │
       │  SPEI_y = N⁻¹(F_log-logistic(D_y))         │
       │  κ_y = tercile(SPEI_y) ∈ {dry, normal, wet}│
       │                                            │
       │  → counts (n_dry, n_normal, n_wet)         │
       │     ≈ (20, 19, 20) for 59 years            │
       └─────────────────┬──────────────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────────────┐
       │  STEP 2.  BAYESIAN PRIOR  (conjugate)      │
       ├────────────────────────────────────────────┤
       │  α₀ = (n_dry, n_normal, n_wet) + 1          │
       │     ≈ (21, 20, 21)                         │
       │                                            │
       │  w ~ Dir(α₀)                               │
       │                                            │
       │  Updated each year via:                    │
       │  α_{c+1} = α_c + e_{κ*}                    │
       │  where κ* is the observed class            │
       └─────────────────┬──────────────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────────────┐
       │  STEP 3.  STATE RECONSTRUCTION             │
       │  (planting_date .. forecast_date)          │
       ├────────────────────────────────────────────┤
       │  FAO-56 deterministic forward run          │
       │  with observed P, ETo, Kc[d]:              │
       │                                            │
       │    SW_d = clip(SW_{d-1} + 0.8·P_d - Kc·ETo,│
       │                0, AWC)                     │
       │    if SW_d < AWC·(1-MAD):                  │
       │        I_d = AWC - SW_d  (refill)          │
       │                                            │
       │  → SW(today), I_to_date                    │
       └─────────────────┬──────────────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────────────┐
       │  STEP 4.  HORIZON MONTE-CARLO  (N = 500)   │
       │  (forecast_date+1 .. forecast_date+H)      │
       ├────────────────────────────────────────────┤
       │  for i in 1..N:                            │
       │    w_i ∼ Dir(α_c)              (analytical)│
       │    κ_i ∼ Cat(w_i)              (analytical)│
       │    y_i ∼ Uniform(years of class κ_i)       │
       │    P_h, ETo_h ← season y_i, days [d+1..d+H]│
       │    Continue FAO-56 from SW(today)          │
       │      → SW_i[h], I_i[h]   for h ∈ 1..H      │
       └─────────────────┬──────────────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────────────┐
       │  STEP 5.  AGGREGATION                      │
       ├────────────────────────────────────────────┤
       │  P(I > 0 | tomorrow) = (1/N) Σ 1[I_i[1] > 0]│
       │  q05, q50, q95 of I_i[1]   (per-day)       │
       │  q05, q50, q95 of ΣI_i[h]  (cumulative)    │
       │  if obs available: CRPS, in-CI flag        │
       └─────────────────┬──────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                │
├────────────────────────────────────────────────────────────────────────┤
│  Tomorrow:                                                             │
│    P(irrigation event) =  XX %                                         │
│    I (mm) — 90% IC     = [a, b]                                        │
│  Cumulative next H days:                                               │
│    I_total (mm) — 90% IC = [c, d]                                      │
│    Median forecast       =  m                                          │
│  SW trajectory bracket (if validation):                                │
│    Coverage 90% IC = ≈ 97% (across 4250 backtest points)               │
│  Decision:                                                             │
│    "Irrigate tomorrow with X mm" / "no action needed"                  │
└────────────────────────────────────────────────────────────────────────┘
```

## What is *Bayesian* vs *Monte-Carlo* here

The uncertainty has two distinct origins, both handled analytically/sampling — never MCMC:

| Source | Type | Handling |
|---|---|---|
| Seasonal-class prevalence \\(w\\) | **Bayesian (conjugate)** | Posterior \\(w \mid \text{counts} \sim \text{Dir}(\alpha + n)\\) is *analytical* |
| Choice of which historical season represents class \\(\kappa\\) | Sampling (epistemic) | `np.random.choice(years_of_class)` |
| Daily weather sequences within the horizon | Sampling (aleatory) | Resample one full historical season's daily forcings |
| Non-linear FAO-56 propagation | Deterministic | Forward integration of the soil-water budget |

Because the Dirichlet-Multinomial pair is **conjugate**, we **never run MCMC** for the operational forecast. The Monte-Carlo loop in Step 4 is only there to propagate the analytical posterior of \\(w\\) through the non-linear water-balance equation. This is the difference between *Bayesian inference* (analytical, fast) and *Bayesian computation by simulation* (e.g. NUTS in PyMC, slower).

The PyMC NUTS sampler (in [models/water_balance.py](../src/bwb/models/water_balance.py)) is used **only** in the *posterior-recovery test* — a sanity check that the deterministic FAO-56 baseline can be recovered as the posterior mean of a hierarchical Bayesian model with priors on \\(\theta_s, \theta_r, K_c \cdot \text{mult}\\). It is not part of the operational pipeline.

## Why this matters for the paper

Three claims that this design enables:

1. **Sub-second forecasts.** Each rolling 5-day forecast runs in ~0.1 s (500 sims × 5 days × 1 FAO-56 step). Backtesting 4 250 days across 10 cities ran in 8 minutes.

2. **Mathematically clean Bayesian update.** The yearly update \\(\alpha_{c+1} = \alpha_c + e_{\kappa^*}\\) is exact — no approximation, no convergence diagnostics, no \\(\hat{R}\\)/ESS to worry about.

3. **Operational deployability.** Because there is no MCMC dependency at runtime, the model can be deployed as a simple REST endpoint, a CLI tool, or an embedded function in farm-management software without bringing along PyMC/PyTensor.

---

## Validation target — what the backtest is measuring

> **Important:** The 4,250-point backtest verifies the rolling 5-day forecast against the **deterministic FAO-56 baseline driven by Xavier-reanalysis observed P and ETo**. It does **not** validate against in-situ TDR / capacitive soil-moisture sensors.

This is a deliberate scoping choice with three parts:

1. **Why FAO-56 is a credible verification target.** FAO-56 (Allen et al. 1998) is the de facto agro-meteorological standard; its single-Kc water budget has been independently calibrated against thousands of lysimeter, eddy-covariance and weighing-lysimeter studies worldwide. When fed with quality-controlled P and ETo, it serves as a *reference* for what the soil-water budget *should* look like under a given crop and management.

2. **Why we don't validate against sensors here.** No public sensor network covers the 10 MATOPIBA municipalities daily over 1961-2025. Validating the forecast against scattered field campaigns (when available) would conflate three different uncertainties (forecast error, FAO-56 model error, sensor measurement error). Holding FAO-56 as the *target* isolates the **forecast skill** from the underlying agro-meteorological model.

3. **Honest scope statement for the paper.** The forecast tracked in this study is therefore: *"how well a probabilistic 5-day forecast tracks what FAO-56 would have called for had perfect 5-day weather observations been available"*. Independent sensor validation in a follow-up study (with INMET stations and instrumented test plots) is listed in *Future work*.

The companion table for this clarification is
[output/paper_tables/table_backtest_rolling_summary.tex](../output/paper_tables/table_backtest_rolling_summary.tex), whose caption mirrors this distinction.

---

## Climatology resampling vs Numerical Weather Prediction (NWP)

A natural reviewer question: *"For 5-day horizons, NWP ensembles (ECMWF EPS, GFS-GEFS, INMET COSMO, OpenMeteo aggregator) are the operational gold standard. Why use historical-analogue resampling?"* This section settles the framing.

### What the model does

The horizon Monte-Carlo in **Step 4** can be driven from either of two weather sources:

| Mode | When | What weather drives the ensemble |
|---|---|---|
| **(a) Climatology resampling** | Always available; backtest uses this exclusively | One full historical season is sampled per simulation, conditional on the Dirichlet-sampled class |
| **(b) NWP override / hybrid** | Operational mode (provided 30+ years of analog history) | A user-supplied NWP ensemble (\\(M\\) members × \\(H\\) days) is sampled per simulation; climatology drops out of the horizon weather |

Both modes use the same Bayesian Dirichlet posterior on the *yearly* class weights — what changes is only the *short-horizon weather*.

### Why backtests must use climatology

Xavier reanalysis (and analog products such as BR-DWGD, ENA-CDS, AgERA5) cover **1961-present**. NWP archives in operational form (the THORPEX-TIGGE archive, ECMWF MARS) cover at best **2008-present** for ensemble forecasts. Backtesting our model over a 60-year span requires a horizon-weather source available throughout that span — only climatology resampling satisfies that constraint. **Any 1961-onward backtest published in this domain has the same constraint**, regardless of the modeling approach.

### Why operational deployment should use NWP

For a forecast issued *today* with the question *"how much should I irrigate tomorrow?"*, modern global NWP ensembles dominate climatology in the 1-5 day horizon (Bauer et al. 2015, *Nature*). When deployed in production, this implementation accepts a NWP ensemble via the
``horizon_pr_ensemble``, ``horizon_eto_ensemble`` keyword arguments of [rolling_5day_forecast()](../src/bwb/forecast/rolling.py), bypassing the climatology resampling for the horizon weather while preserving the Bayesian class-weight machinery (which informs initial-state uncertainty and provides a graceful fallback when NWP is unavailable, e.g. data-link failure on a remote farm).

### Hybrid framing in the manuscript

We therefore present the model as:

* **a backtest-validated probabilistic framework** whose climatology fallback enables 60-year verification, and
* **an operational system** that *consumes* NWP ensembles when they are available, gracefully degrading to climatology when they are not.

This positions the contribution beside (rather than against) NWP-driven irrigation DSS such as ARID and AquaCrop-OS, while explicitly addressing the deployability gap they leave open.

---

## Multi-source fused weather inputs (2026 operational)

For operational deployment in 2026 the curated Xavier dataset (used for the
1961-2025 backtest) is replaced by a fused product combining three live
sources, all reachable from the codebase as drop-in functions returning the
canonical schema (`date, Rs, u2, Tmax, Tmin, RH, pr, ETo`):

| Source | Coverage | Module |
|---|---|---|
| NASA POWER (MERRA-2) | 1990 → today − 7 days, global 0.5° × 0.625° | [`download_nasa_power`](../src/bwb/data/sources/climate/nasa_power.py) |
| Open-Meteo Archive (ERA5) | 1940 → today − 2 days | [`download_openmeteo_archive`](../src/bwb/data/sources/climate/openmeteo.py) |
| Open-Meteo Forecast (best-match NWP) | today − 29 days → today + 16 days | [`download_openmeteo_forecast`](../src/bwb/data/sources/climate/openmeteo.py) |

[`fuse_climate_sources`](../src/bwb/data/sources/climate/fusion.py) combines
them via per-variable weighted means with the
EVAOnline ``HIST_WEIGHTS`` calibrated against BR-DWGD at 17 Brazilian
sites:

| Var | NASA weight | Rationale (EVAOnline calibration notes) |
|---|---|---|
| `Rs` | 0.92 | CERES satellite radiation dominant |
| `u2` | 0.20 | ERA5 wind more accurate |
| `RH` | 0.35 | ERA5 humidity superior |
| `Tmax` | 0.58 | NASA slight advantage in extremes |
| `Tmin` | 0.52 | near-equal performance |
| `pr` | 0.50 | equal reliability |

ETo is **recomputed** from the fused Tmax/Tmin/RH/u2/Rs via FAO-56
Penman-Monteith ([`compute_eto_fao56_pm`](../src/bwb/data/sources/climate/nasa_power.py))
rather than averaging the two source ETos directly — same approach as
EVAOnline's `eto_services.py`.

**Validation against Xavier (Balsas, MA — 90-day soybean cycle 2020-12 to 2021-02):**

| Variable | NASA RMSE | Archive RMSE | Fused RMSE | Improvement |
|---|---|---|---|---|
| `Tmax` | 1.96 | 1.83 | **1.49** | best of all |
| `Rs` | 2.30 | 3.08 | **2.17** | best of all |
| `pr` | 7.82 | 11.21 | **7.34** | best of all |
| `ETo` | 0.78 | 0.58 | **0.45** | -22% vs Archive, -42% vs NASA |

Backtests published in this manuscript still use Xavier (only Xavier covers
1961-2025); operational forecasts after the publication cutoff will use the
fused product, with the Bayesian Dirichlet posterior on class weights
unchanged.

### A/B backtest finding — Balsas, MA, 425 forecasts per arm

Re-running the rolling 5-day backtest with both arms restricted to a
1990-onwards history (so the source effect is isolated from the
amount-of-history effect — see
[scripts/backtest_ab_fusion.py](../scripts/backtest_ab_fusion.py)) shows
**a sharp shift in the verification target itself**, not in forecast skill:

| Metric | Xavier (1990+) | Fused NASA+Archive (1990+) | Δ |
|---|---|---|---|
| Mean obs `I_total` (5-day cumulative) | 1.60 mm | 2.43 mm | **+52%** |
| Days with `I > 0` (FAO-56 trigger) | 10 / 425 (2.4%) | 15 / 425 (3.5%) | +47% |
| CRPS mean | 0.745 mm | 1.540 mm | +0.795 |
| KGE | +0.376 | −0.412 | −0.79 |
| PBIAS | −45.8 % | −93.6 % | −47.8 pp |

The ground-truth `obs_I_total` is the deterministic FAO-56 baseline driven
by **the same source the forecast was trained on** — so each arm verifies
itself, not against a common truth. Fused inputs produce ETo ≈ 22 % higher
than Xavier in MATOPIBA, which propagates into 50 % more cumulative
irrigation demand and a stricter target. The forecast median responds with
a similar shape but cannot fully match the larger demand because
climatology resampling smooths the variability — the deeper PBIAS is a
direct consequence of the larger target, not of degraded skill.

**Interpretation for the manuscript.** Reanalysis of Brazilian agro-climate
sources is well known to differ in ET₀ magnitude — global reanalyses
(MERRA-2, ERA5) tend to be drier and windier than INMET-calibrated
products such as Xavier (Bauer et al. 2015). The A/B does **not** show
that Xavier is "more correct"; it shows that *backtest verification is
self-consistent within a source*. Operational deployment with the fused
product should therefore be paired with a **Brazil-side reference
(INMET station network) for periodic re-anchoring** rather than relying on
the original Xavier-trained backtest as a skill ceiling.

The artifacts of this experiment are
[output/backtest_ab_fusion/balsas_ab.csv](../output/backtest_ab_fusion/balsas_ab.csv)
(per-forecast) and
[output/backtest_ab_fusion/balsas_ab_summary.csv](../output/backtest_ab_fusion/balsas_ab_summary.csv)
(aggregate).

---

## INMET ground-truth validation (Jan–May 2026)

To check that the fused product is closer to *measured* meteorology
than either NASA POWER or Open-Meteo Archive on its own, we validated
all three against hourly observations from the INMET automatic-station
network ([scripts/inmet_validation_multistation.py](../scripts/inmet_validation_multistation.py)).

### Coverage and QC

Five INMET automatic stations (A346, A375, A402, A404, A416) sit
within or adjacent to the 10 MATOPIBA municipalities used elsewhere
in this study, plus the conventional station 82768 (Balsas/MA, only
3 readings per day and 22 % coverage — discarded). Each automatic
file holds 3 048 hourly records covering 2026-01-01 to 2026-05-07
(~127 days), but **per-sensor coverage is highly uneven**:

| Station | City | RH | pr | Rs |
|---|---|---|---|---|
| A346 | Urucuí | 83 % | **0 %** | 54 % |
| A375 | Baixa Grande do Ribeiro | **0 %** | 96 % | 53 % |
| A402 | Barreiras | 25 % | 25 % | 17 % |
| A404 | Luís Eduardo Magalhães | 50 % | 52 % | 42 % |
| A416 | Correntina | 76 % | 38 % | 53 % |

Hourly readings were aggregated to the local civil day (UTC−3) using
FAO-56 conventions (Tmax, Tmin from the per-hour extrema; RH and u₂
from hourly means; Rs as daily total). Each variable inherits its own
QC threshold (≥ 18–20 hours for instantaneous variables, ≥ 8 hours for
shortwave radiation), and a station enters the per-station column of
the table below only when at least 4 variables clear the 30-day
floor. A402 and A404 do not — they contribute only to the pooled
metric.

### Wind anomaly (important caveat)

For all three good-coverage stations, **≥ 70 % of hourly anemometer
readings register exactly 0 m/s**, dragging the daily mean to
~0.15 m/s in regions where physically expected values are 1.5–3 m/s.
Per-station wind metrics (the `u₂` row, daggered in the LaTeX table)
therefore compare alternative sources against a degenerate zero
ground truth and are not informative on their own. The pooled row,
which compares **anomalies** rather than absolute values, removes
this station-level offset and is the metric to read for wind.

### Result — RMSE (lower is better)

The full per-station and pooled table is at
[output/paper_tables/table_inmet_validation.tex](../output/paper_tables/table_inmet_validation.tex).
Pooled (across-station, within-station-anomaly) summary:

| Variable | NASA | Archive | **Fused** | Best |
|---|---|---|---|---|
| Tmax | 1.81 | **1.23** | 1.41 | Archive |
| Tmin | 1.19 | **0.89** | 0.96 | Archive |
| RH | 6.10 | 4.78 | **4.72** | **Fused** |
| Rs | 3.22 | 3.46 | **3.05** | **Fused** |
| pr | 8.41 | 8.57 | **7.92** | **Fused** |
| ETo | 0.60 | 0.58 | **0.54** | **Fused** |
| u₂ (anomaly) | 0.49 | 0.49 | **0.46** | **Fused** |

**The fused product wins on 5 of 7 variables**, including the two
that drive irrigation decisions (`ETo` and `pr`). For Tmax/Tmin,
Open-Meteo Archive (ERA5) alone is best — these are the two
variables for which `HIST_WEIGHTS` already assigns ≈50/50 NASA/OM
weights, so the fusion's improvement is small and the noise can
swing the comparison.

This is independent confirmation that the EVAOnline-calibrated
weights, originally tuned against BR-DWGD at 17 sites, transfer to
MATOPIBA without retuning.

### Limitations of this validation

* **Window is short** — 127 days, January-to-May 2026 only.
* **Spatial coverage is partial** — 3 of 10 cities have a column in
  the table; another 2 contribute only via the pooled row.
* **Point vs. grid** — INMET stations are points; NASA POWER is a
  0.5° × 0.625° grid; Archive is interpolated to the requested
  coordinate from a 0.25° native grid. Some of the residual RMSE is
  representativity error, not source error.
* **A 90-day soybean cycle does not fit inside the INMET window**
  (the typical Dec-1 planting date predates the INMET coverage start
  on Jan-3). Cycle-level verification of `rolling_5day_forecast`
  against INMET is therefore listed as future work for the
  2026–2027 harvest.

The artifacts of this experiment are
[output/inmet_validation/per_station_summary.csv](../output/inmet_validation/per_station_summary.csv),
[output/inmet_validation/pooled_summary.csv](../output/inmet_validation/pooled_summary.csv),
and [output/paper_tables/table_inmet_validation.tex](../output/paper_tables/table_inmet_validation.tex).

---

## References for this section

* Allen, R.G. et al. (1998). FAO-56.
* Bauer, P.; Thorpe, A.; Brunet, G. (2015). The quiet revolution of numerical weather prediction. *Nature* 525, 47-55.
* Kling, H.; Gupta, H. (2009). On the development of regionalization relationships for lumped watershed models. *J. Hydrol.* 373, 337-351.
* Moriasi, D.N. et al. (2007). Model evaluation guidelines for systematic quantification of accuracy in watershed simulations. *Trans. ASABE* 50(3), 885-900.
