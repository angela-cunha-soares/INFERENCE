# Methodology

This page documents the design choices that justify the headline numbers
reported in the README. Read it alongside [`limitations.md`](limitations.md).

## Two complementary uses of the framework

The bwb framework supports **two distinct uses** that should not be confused:

| Use | CLI command | Purpose | Reference |
|---|---|---|---|
| **Posterior-recovery test** | `bwb posterior-recovery` (legacy: `validate`) | Internal consistency: does the Bayesian model recover the parameters that generated the FAO-56 reference trajectory? | (none -- self-test) |
| **Climatological sequential forecast** | `bwb forecast-sequential` | Production validation against Xavier reanalysis. Forecast each cycle using only past climatology, then update the prior with the realised cycle. | (this section) |

The headline 150-combination KGE = 0.91 reported earlier comes from the
**posterior-recovery test** -- it verifies that the PyMC implementation
faithfully reproduces FAO-56 behaviour. It is not a forecast skill score.

The numbers that should be quoted in the manuscript as **predictive skill**
are produced by `bwb forecast-sequential`, described below.

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
realised 2020/21 through 2023/24 cycles enter the training set as
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
prior:      Dir(alpha_c)                       # initialised Dir(1, 1, 1)
counts:     n_dry, n_normal, n_wet of training cycles
posterior:  Dir(alpha_c + n_counts)            # closed-form, no MCMC needed
```

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

### Step 5 -- Evaluation against the realised cycle

The realised cycle (Xavier P, ETo for cycle c) is fed into the same
FAO-56 deterministic baseline to obtain the observed SW_obs and I_obs.
Skill scores:

* **Deterministic**: KGE_SW, NSE_SW, MAE_SW, MAE_I, I_total error
* **Probabilistic**: CRPS on I_total; coverage_90 of SW; PIT of I_total

### Step 6 -- Sequential update

The realised cycle is classified using the *training-set* tercile
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

## Reproducibility

```bash
python -m pip install -e ".[parallel]"

# Posterior-recovery test (internal consistency)
bwb posterior-recovery

# Production validation: sequential forecast 2020..2024
bwb forecast-sequential --cities Balsas Barreiras --cycles 2020 2021 2022 2023 2024

# Operational forecast 2025/26 using the alpha posterior from above
bwb forecast-sequential --cycles 2025 \
    --alpha-init-from output/forecast_sequential/alpha_final_Balsas.json
```

Outputs:

* `output/forecast_sequential/forecast_<city>_<year>_<year+1>.csv`
* `output/forecast_sequential/observed_<city>_<year>_<year+1>.csv`
* `output/forecast_sequential/sequential_summary_all_cities.csv`
* `output/forecast_sequential/alpha_final_<city>.json`
