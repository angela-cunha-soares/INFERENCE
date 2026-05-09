# bwb -- Bayesian Water Balance framework

A hierarchical Bayesian framework for stochastic water-balance modelling and
risk-aware irrigation scheduling, validated across the **MATOPIBA**
agricultural frontier (Maranhao, Tocantins, Piaui, Bahia -- Brazil).

The framework couples the FAO-56 single-crop-coefficient process equation
with a state-space Bayesian observation model implemented in PyMC v5 / PyTensor,
and propagates parametric + meteorological uncertainty into probabilistic
irrigation recommendations.

## Highlights

- FAO-56 daily water balance with per-day Kc curves for 90/100/120-day soybean
- Hierarchical priors on theta_s, theta_r, Kc-multiplier and theta_init informed by
  WMO climatological normals (1961-1990, 1981-2010, 1991-2020) and a
  city x variable x month distribution atlas
- Climatic indices (SPEI, IA, terciles, ENSO ONI/MEI) as covariates / priors
- Probabilistic decision rule with calibrated 90% credible intervals
  (coverage_90 ~ 0.90, KGE ~ 0.88, NSE ~ 0.97 on the validation grid)
- Sequential Bayesian updating across the growing season
- Full validation grid: 10 MATOPIBA cities x 3 root depths x 5 crop cycles
- Sobol' global sensitivity analysis
- Reproducible CLI (`bwb info / validate / sensitivity / forecast / figures`)

## Project layout

```
INFERENCE/
+-- src/bwb/                   Python package (installed as `bwb`)
|   +-- cli.py                 Subcommand entry point
|   +-- config/                Settings + regional TOML profiles
|   +-- data/                  Loaders & adapters (city CSV, oceanic, Xavier)
|   +-- decision/              Probabilistic irrigation rules
|   +-- forecast/              Nested-MC ensemble propagator (GEFS x posterior)
|   +-- inference/             MCMC, VI, diagnostics, posterior predictive
|   +-- models/                FAO-56 water balance, Van Genuchten retention
|   +-- phenology/             Crop / Kc curves
|   +-- priors/                Climatological + oceanic + indices (SPEI, IA)
|   +-- rolling/               Sequential Bayesian updates
|   +-- utils/                 Path resolution, logging
|   +-- validation/            Pipeline, metrics, sensitivity analysis
+-- scripts/                   Orchestration scripts (validation, sensitivity, ...)
+-- tests/                     pytest suite (unit + integration)
+-- data/                      Curated input data (Balsas, 10 cities, FAO Kc)
+-- data_raw/                  Raw downloads (Xavier NetCDF, oceanic indices)
+-- data_processed/            Climatological normals, distribution atlas
+-- output/                    Run outputs (CSV, NetCDF traces, figures)
+-- figures/                   Publication-ready figures
+-- notebooks/paper/           Reproducible analysis notebooks
+-- docs/                      MkDocs site
```

## Quick start

```bash
# 1. Install (editable mode for development)
python -m pip install -e .

# 2. Inspect runtime settings + active profile
bwb info

# 3. Smoke-test on a single combination (~30 s)
bwb validate --fast --cities Balsas --depths 60 --cycles 2023

# 4. Full validation grid (10 cities x 3 depths x 5 cycles ~ 20 min on 4 cores)
python -m pip install joblib
bwb validate --jobs 4

# 5. Sobol' global sensitivity analysis (~30 s)
bwb sensitivity --city Balsas --cycle 2023 --n 1000

# 6. Probabilistic forecast for one cycle (Bayes fit + GEFS-like ensemble)
bwb forecast --city Balsas --cycle 2023 --members 31 --posterior-draws 200

# 7. Regenerate publication figures
bwb figures
```

Outputs land in `output/validation/`, `output/sensitivity/`, `output/forecast/`,
and `figures/paper/` respectively.

## Configuration

All runtime defaults live in `src/bwb/config/regional/<region>.toml`. The
shipped `matopiba.toml` profile encodes:

- 10 city coordinates
- FAO-56 90-day soybean cycle (planting Dec 1)
- Sandy-loam reference soil (SoilGrids texture for the region)
- Calibrated Bayesian priors (`sigma_obs_max = 0.10`, `sigma_theta_s = 0.04`,
  `sigma_theta_init = 0.06`)

Override via env vars (`BWB_DRAWS`, `BWB_OUTPUT_DIR`, `BWB_REGION`, ...) or
pass `--profile <name>` to the CLI.

## Data dependencies

| Dataset                                    | Source             | Status                                   |
| ------------------------------------------ | ------------------ | ---------------------------------------- |
| Daily climate (1961-2025) for 10 cities    | Xavier reanalysis  | shipped                                  |
| Balsas reference series                    | INMET / Xavier     | shipped                                  |
| FAO-56 crop library                        | Allen et al. 1998  | shipped (`data/crops/`)                  |
| Climatological normals (3 WMO periods)     | computed locally   | shipped (`data_processed/`)              |
| Distribution atlas                         | computed locally   | shipped (`data_processed/`)              |
| ONI (1961-2025)                            | NOAA / PSL         | shipped                                  |
| MEI raw                                    | NOAA / PSL         | shipped, *not yet consolidated*          |
| AMM, AMO, PDO, MJO, IOD                    | NOAA               | *pending download*                       |
| GEFS reforecast ensemble                   | NOAA NCEI          | *pending* (synthetic fallback available) |
| Soil-moisture reference (GLEAM/ESA CCI)    | GLEAM v3.8 or ESA CCI SM | *pending* (Xavier does not include soil moisture) |

Use `scripts/extract_xavier_matopiba.py`, `scripts/download_oni.py`,
`scripts/download_mei.py` and `scripts/oceanic_indices_consolidator.py` to
regenerate the processed artefacts when the raw inputs are refreshed.

## Validation snapshot

The framework provides three complementary validation modes (A, B, C
below).

### A. Posterior-recovery test (150 combinations)

Full retrospective grid: **10 cities x 3 root depths x 5 crop cycles =
150 combinations**, all converged. Run `bwb validate` (or
`scripts/analyze_validation.py` for the post-hoc summary).

Aggregate metrics across the 150 successful fits:

| metric            | mean  | median | std   | min   | max   |
| ----------------- | ----- | ------ | ----- | ----- | ----- |
| RMSE              | 0.009 | 0.008  | 0.003 | 0.004 | 0.016 |
| MAE               | 0.007 | 0.006  | 0.002 | 0.003 | 0.015 |
| Bias              | 0.000 | 0.000  | 0.000 | 0.000 | 0.001 |
| KGE               | 0.909 | 0.918  | 0.043 | 0.798 | 0.985 |
| NSE               | 0.976 | 0.979  | 0.012 | 0.933 | 0.997 |
| CRPS              | 0.005 | 0.005  | 0.002 | 0.002 | 0.010 |
| alpha-PIT         | 0.894 | 0.902  | 0.047 | 0.749 | 0.972 |
| coverage_90       | 0.918 | 0.922  | 0.027 | 0.856 | 0.978 |
| interval_score_90 | 0.039 | 0.037  | 0.012 | 0.014 | 0.077 |

*All 150 combinations converged with R-hat below 1.05; coverage of the
90% credible interval is well calibrated to nominal.*

### B. Climatological sequential forecast (50 combinations)

Strictly causal forecast (training on 1961..c-1, evaluation on cycle c)
across **10 cities x 5 cycles (2020/21..2024/25) = 50 combinations**,
initialised with a climatologically-informed Dirichlet prior on the
SPEI-tercile counts of the 1961-2019 training window plus a unit
smoother. Run `bwb forecast-sequential --n-sim 500` or
`python scripts/generate_forecast_seq_tables.py` for the LaTeX tables.

Aggregate skill across the 50 forecast cycles:

| metric                                | mean  | median |
| ------------------------------------- | ----- | ------ |
| CRPS on seasonal irrigation (mm)      | 20.63 | 18.18  |
| Coverage_90 daily soil-water content  | 0.954 | 0.956  |
| Coverage_90 seasonal irrigation depth | 0.96  | 1.00   |
| KGE on daily soil-water content       | -0.10 | -0.09  |
| PIT on seasonal irrigation depth      | 0.46  | 0.41   |
| CRPSS vs naive climatological mean    | +0.03 | +0.04  |

*Climatological forecast intervals are conservatively calibrated; the
KGE on the daily trajectory is negative by design (a climatological
forecast does not aim to predict the realised daily sequence).*

### C. Operational rolling 5-day forecast (4,250 forecasts)

Per-city verification of `bwb backtest-rolling` against the FAO-56
deterministic baseline (Xavier observed P and ETo) across day 0
through day 84 of the five 2020-2024 cycles for the ten MATOPIBA hubs:
**N = 4,250 forecasts** in aggregate. Median KGE +0.32, mean CRPS
2.65 mm, coverage_90 = 0.970.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).
