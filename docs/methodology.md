# Methodology

This page documents the design choices that justify the headline numbers
reported in the README. Read it alongside [`limitations.md`](limitations.md)
and [`methodology_diagram.md`](methodology_diagram.md).

## Three complementary validation modes

The bwb framework supports **three distinct validation modes** that should
not be confused. Each answers a different question and is exposed as a
separate CLI subcommand.

| Mode | CLI command | Purpose | What it answers |
|---|---|---|---|
| **A. Posterior-recovery test** | `bwb posterior-recovery` (legacy: `validate`) | Internal consistency. The Bayesian model is given a synthetic FAO-56 trajectory and must recover the parameters that generated it. | "Does the PyMC implementation reproduce the FAO-56 process within calibrated uncertainty?" |
| **B. Climatological sequential forecast** | `bwb forecast-sequential` | Strictly causal seasonal forecast (training on 1961..c-1, forecast on cycle c). | "What is the seasonal irrigation distribution before any data from cycle c is observed?" |
| **C. Operational rolling 5-day forecast** | `bwb backtest-rolling` | 5-day-ahead probabilistic forecast of cumulative irrigation depth from any day-of-cycle. | "How much irrigation will the crop need over the next 5 days, with what credible interval?" |

The headline 150-combination KGE = 0.918 (median) and 90% coverage = 0.922
come from mode **A**: it verifies that the PyMC implementation faithfully
reproduces FAO-56 behavior. It is not a forecast skill score.

The headline forecast skill scores (mean CRPS = 20.6 mm, coverage = 0.954,
CRPSS = +0.034 against the naive climatology) come from mode **B**, which
is the production validation against Xavier reanalysis.

The headline operational scores (median KGE = +0.32, mean CRPS = 2.65 mm,
coverage = 0.970) come from mode **C**, which is the deployment-facing
mode for day-to-day decisions.

## Process layer (FAO-56 daily water balance)

```
SW_d = clip(SW_{d-1} + P_eff_d - ETc_d, 0, AWC)
ETc_d = Kc_d x ETo_d
P_eff_d = 0.8 x P_d                                # FAO-56 simplified
if SW_d < AWC * (1 - MAD): irrigate to AWC         # decision rule
```

For early-cycle soybean (90 days, planted 1 December) the reference Kc
curve from FAO-56 Table 12 / FAO-66 is:

| Stage          | Days | K_c    |
|----------------|------|--------|
| Initial        | 15   | 0.40   |
| Development    | 15   | 0.80   |
| Mid-season     | 40   | 1.15   |
| Late           | 15   | 0.80   |
| Harvest        | 5    | 0.50   |
| **Total**      | 90   | --     |

Effective rooting depth Z_r = 60 cm; AWC = 120 mm; MAD = 0.55 (FAO-56
default for soybean).

## Climatological sequential forecast (Section 4 of the paper)

### Step 1 -- Training set per target cycle

For target cycle c (planting year c, harvest year c+1):

```
Training set = all complete cycles in 1961..(c - 1)
```

For example, the 2020/21 target uses **59 historical cycles (1961-2019)**;
the 2024/25 target uses **63 historical cycles (1961-2023)** -- the
realized 2020/21 through 2023/24 cycles enter the training set as
they become observed.

### Step 2 -- Per-cycle SPEI classification

For each historical cycle compute the seasonal water-balance D = P - ETo.
Standardise via SPEI (Vicente-Serrano et al. 2010):

```
1) shift D so all values > 0
2) fit log-logistic (Fisk) by MLE
3) SPEI = N^{-1}(F_LL(D))
```

Then partition the historical SPEI distribution into terciles:

```
SPEI <= q33    -> dry
q33 < SPEI <= q67  -> normal
SPEI >  q67    -> wet
```

This combines the SPEI standardisation (Vicente-Serrano 2010) with the
quantile classification of Pinkayan (1966) and Xavier et al. (2002).

### Step 3 -- Conjugate Dirichlet-Multinomial update

```
prior:      Dir(alpha_c)                       # default: climatologically-informed
                                               # alpha_c = n_(1961-2019) + 1 (smoother)
                                               # ablation: Dir(1, 1, 1) uniform
counts:     n_dry, n_normal, n_wet of training cycles
posterior:  Dir(alpha_c + n_counts)            # closed-form, no MCMC needed
```

The default initial hyperparameter is the **climatologically-informed**
prior `alpha_2020 = n_(1961-2019) + 1` (typically `n ≈ (20, 20, 19)` by
construction of equal-frequency terciles). The uniform-prior variant
`alpha_2020 = (1, 1, 1)` is exposed for ablation purposes through
`--alpha-init 1 1 1`; on the 50-combination validation grid it inflates
the median CRPS from 18.2 mm to ≈ 23 mm because the five available target
cycles are not enough to refine an uninformative posterior. The choice of
prior is therefore non-trivial and is documented in the manuscript
Discussion (Section "The role of the priors in the reported skill").

### Step 4 -- Monte-Carlo forecast

For each of N = 500 simulations:

```
1) sample weights w ~ Dir(alpha_post)
2) sample category k ~ Categorical(w)
3) sample uniformly one historical cycle from class k
4) run the FAO-56 daily water balance with that cycle's (P, ETo)
   -> trajectory (SW_d, I_d, DP_d, ETc_d) for d = 1..90
```

The aggregated distribution gives forecast quantiles SW_q05, SW_q50,
SW_q95 and probabilistic irrigation depth I_q05, I_q50, I_q95.

### Step 5 -- Evaluation against the realized cycle

The realized cycle (Xavier P, ETo for cycle c) is fed into the same
FAO-56 deterministic baseline to obtain the observed SW_obs and I_obs.
Skill scores:

* **Deterministic**: KGE_SW, NSE_SW, MAE_SW, MAE_I, I_total error
* **Probabilistic**: CRPS on I_total; coverage_90 of SW; PIT of I_total

### Step 6 -- Sequential update

The realized cycle is classified using the *training-set* tercile
thresholds (no retraining), and its observed class increments alpha:

```
alpha_{c+1} = alpha_c + e_{observed_class}    # one-hot
```

The cycle 2025/26 forecast (operational) uses alpha after observing
2020/21 through 2024/25.

### Honest information flow

| Quantity used to forecast cycle c | Earliest origin |
|---|---|
| SPEI thresholds (terciles)          | Refit on 1961..(c-1) |
| Dirichlet alpha                     | alpha_init + observations from cycles < c |
| Resampled (P, ETo)                  | One historical cycle in 1961..(c-1) |
| K_c curve                           | FAO-56 Table 12 (literature, not data-derived) |
| AWC, MAD                            | FAO-56 / Van Genuchten (literature) |

**No information from cycle c (or later) is ever used to forecast cycle c.**

## The framework is global; the validation is regional

| Layer | Status |
|---|---|
| Architecture (PyMC, profiles, loaders, decision rule) | global |
| FAO-56 crop library | 16 categories, hundreds of cultivars (FAO Table 12) |
| Soil hydraulics (Van Genuchten + textural classes) | global, with on-demand SoilGrids client (planned) |
| Climate forcings | any gridded daily P/ETo product (Xavier for Brazil; AgERA5 / Princeton globally) |
| Regional profile | shipped: matopiba.toml; trivially extensible to any region (cities, planting calendar, soil texture) |
| **Validation** | 10 MATOPIBA cities x soybean 90-day x 5 cycles (2020-2024); operational forecast 2025/26 |

To validate in a new region, create `<region>.toml` with city
coordinates, planting calendar and soil texture; the rest of the
pipeline is region-agnostic.

## DAG learning over the climate inputs

In addition to the FAO-56 process layer, the framework exposes a
data-driven Bayesian-network learner
(`bwb.inference.dag_learning`) that recovers the conditional-dependency
structure between the daily climate variables (Tmax, Tmin, Tmean, RH,
u2, Rs, P) and ETo. The methodological precedent is Ribeiro et al.
(IJCNN 2022), who learnt a Bayesian network for ETo over five INMET
stations of MATOPIBA and identified u2 and minimum relative humidity as
parents of ETo. We adopt the same toolchain (HillClimb + BDeu via
[`pgmpy`](https://pgmpy.org)), broaden the input set and the population
(pooled 65-year Xavier reanalysis across the ten MATOPIBA hubs,
n ≈ 237,410), and require **agreement between BDeu and BIC scorers** on
the consensus graph. The recovered direct parents of ETo are
`{R_s, T_min, u_2}`, with `P` correctly identified as conditionally
independent of ETo.

Run with: `python scripts/learn_climate_dag.py`. Outputs in
`figures/paper/fig_climate_dag.png` and
`output/paper_tables/table_dag_relevance.{csv,tex}`.

## ENSO-conditioned dynamic Bayesian network

The Dirichlet posterior of the cycle-level forecast treats yearly class
draws as exchangeable. To encode the dependency on the El Niño/Southern
Oscillation regime, the framework also exposes a dynamic Bayesian
network (`bwb.models.dbn_ensoclass`) in which the season class depends
on the seasonal ENSO regime classified from the 3-month running ONI.
Sampled with NUTS (4 chains, 1,000 draws). The ENSO conditioning
sharpens the dry-class probability under El Niño from the static 0.34 to
0.38 and shifts the neutral regime towards the wet class.

Run with: `python scripts/fit_dbn_enso.py`. Outputs in
`output/state_space/trace_dbn_enso.nc`,
`figures/paper/fig_dbn_enso.png`, and
`output/paper_tables/table_dbn_enso.{csv,tex}`.

## Operational fused climate product

Because the curated Xavier reanalysis is not updated in real time, the
operational deployment substitutes a **multi-source fused product**
combining NASA POWER (MERRA-2), Open-Meteo Archive (ERA5), and
Open-Meteo Forecast (best-match NWP). Per-variable weights are
calibrated against BR-DWGD at 17 Brazilian sites and validated against
in-situ INMET automatic stations on the MATOPIBA hubs (see
`docs/methodology_diagram.md` and the manuscript Appendix A). ETo is
recomputed by Penman-Monteith on the fused inputs rather than averaged
from the per-source ETo.

Modules: `bwb.data.sources.climate.{nasa_power, openmeteo, fusion}`.

## Reproducibility

```bash
python -m pip install -e ".[parallel]"

# A. Posterior-recovery test (internal consistency, 150 combinations)
bwb posterior-recovery

# B. Production validation: sequential forecast 2020..2024
bwb forecast-sequential --cities Balsas Barreiras --cycles 2020 2021 2022 2023 2024

# B'. Operational forecast 2025/26 using the alpha posterior from above
bwb forecast-sequential --cycles 2025 \
    --alpha-init-from output/forecast_sequential/alpha_final_Balsas.json

# C. Operational rolling 5-day forecast (4,250 forecasts)
bwb backtest-rolling
python scripts/analyze_backtest_rolling.py
```

Outputs:

* **Mode A**: `output/validation/validation_<TS>.{csv,parquet}`
* **Mode B**: `output/forecast_sequential/forecast_<city>_<year>_<year+1>.csv`,
  `observed_<city>_<year>_<year+1>.csv`,
  `sequential_summary_all_cities.csv`,
  `alpha_final_<city>.json`
* **Mode C**: `output/backtest_rolling/backtest_rolling_h5d.csv`,
  `output/paper_tables/table_backtest_rolling_summary.{csv,tex}`,
  `figures/paper/backtest_rolling_*.png`
