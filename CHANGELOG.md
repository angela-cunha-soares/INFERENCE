# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Climatologically-informed Dirichlet prior** for the sequential forecast
  (`src/bwb/forecast/climatological.py`): alpha_2020 = SPEI-tercile counts on
  1961-2019 plus a unit smoother. The previous uniform-prior outputs are
  archived under `output/forecast_sequential_uniform_prior_archive/` for
  ablation reproducibility.
- **Operational rolling 5-day forecast** (`src/bwb/forecast/rolling.py`,
  `bwb backtest-rolling`) with ten-city verification table
  (`output/paper_tables/table_backtest_rolling_summary.{csv,tex}`) and four
  paper figures (`figures/paper/backtest_rolling_*.png`).
- **DAG structure learning over the climate inputs** (`src/bwb/inference/
  dag_learning.py`, `scripts/learn_climate_dag.py`,
  `figures/paper/fig_climate_dag.png`,
  `output/paper_tables/table_dag_relevance.{csv,tex}`) and Manuscript
  Appendix B (`manuscript/appendix_dag.tex`).
- **ENSO-conditioned dynamic Bayesian network** (`src/bwb/models/
  dbn_ensoclass.py`, `scripts/fit_dbn_enso.py`,
  `output/state_space/trace_dbn_enso.nc`,
  `figures/paper/fig_dbn_enso.png`,
  `output/paper_tables/table_dbn_enso.{csv,tex}`).
- **Multi-source fused climate product** (NASA POWER + Open-Meteo Archive,
  `src/bwb/data/sources/climate/{nasa_power,openmeteo,fusion}.py`) with
  in-situ INMET validation across five MATOPIBA stations
  (`scripts/inmet_validation_multistation.py`,
  `output/paper_tables/table_inmet_validation.tex`,
  Manuscript Appendix A).
- **Reduced-input ETo alternatives benchmark** (Hargreaves-Samani vs.
  Benavides-Lopez vs. Penman-Monteith on the Xavier 1961-2025 series:
  `scripts/eto_alternatives_comparison.py`,
  `output/paper_tables/table_eto_alternatives.{csv,tex}`,
  `figures/paper/fig_eto_alternatives.png`).
- **MCMC convergence diagnostics table** for the state-space model
  (`scripts/generate_mcmc_diagnostics.py`,
  `output/paper_tables/table_mcmc_diagnostics.{csv,tex}`).
- **Methodology flowchart** (`scripts/generate_methodology_flowchart.py`,
  `figures/paper/methodology_flowchart.{png,pdf}`).
- **Operational notebook** (`notebooks/operational_irrigation_5day.ipynb`)
  demonstrating the rolling 5-day forecast end-to-end (gitignored due to
  output cell size).
- Helper scripts `scripts/generate_forecast_seq_tables.py` and
  `scripts/check_tex_refs.py`.
- Bib entries `Kling2012`, `Moriasi2007` to support the rolling-backtest
  table.

### Changed
- Manuscript ([manuscript/manuscript.tex](manuscript/manuscript.tex))
  refreshed end-to-end: abstract, Eq.~(5), Tables `tab:forecast_seq`,
  `tab:forecast_by_city`, `tab:forecast_by_cycle`, Discussion and
  Conclusion now reflect the climatologically-informed prior. The
  sequential-forecast median CRPS dropped from 23.0 mm (uniform prior)
  to 18.2 mm (informed prior); coverage on daily soil-water content
  remains 0.954 (mean) at the nominal 0.90 level.
- `output/paper_tables/table_backtest_rolling_summary.tex`: removed
  invalid `\citet{methodology_diagram}` reference.
- `.gitignore`: added `data_raw/inmet/`, `output/_exploratory/`,
  `output/inmet_validation/_cache/`, and the operational notebook.
- The exploratory single-city A/B fusion benchmark
  (`scripts/backtest_ab_fusion.py`) was archived under
  `output/_exploratory/backtest_ab_fusion/`; a systematic ten-city
  assessment is deferred to a companion deployment study.

- Complete `src/bwb/` package structure with 17 populated modules
- Bayesian inference modules: `mcmc.py`, `diagnostics.py`, `ppc.py`, `vi.py`
- Decision module: `utility.py` with probabilistic irrigation rules
- Rolling sequential updates: `sequential.py` for seasonal Bayesian updates
- Validation framework: `pipeline.py`, `sensitivity.py`, `metrics.py`
- Phenology modules: `crop.py`, `kc_curves.py` for FAO-56 crop coefficients
- Soil hydrology: `van_genuchten.py` with retention curve models
- Prior identification: `identification.py`, `base.py` for distribution priors
- Climatic indices module: `priors/indices.py` with SPEI (non-parametric
  Vicente-Serrano), IA (UNEP 1992), tercile classification (Pinkayan 1966)
  and seasonal-summary helpers
- Forecast ensemble: `forecast/ensemble.py` with `propagate_ensemble`,
  `posterior_from_idata`, and a synthetic GEFS-like ensemble fallback
- Data loaders/adapters: `data/loaders.py` and `data/adapters.py` for
  per-city CSV, oceanic indices, climatological priors, distribution atlas
- Settings + profile loader: `config/settings.py` (env-var driven dataclass)
  and `config/profiles.py` (TOML loader); `matopiba.toml` regional profile
- CLI entry-point `bwb` with subcommands `info / validate / sensitivity /
  forecast / figures`
- Orchestration scripts: `run_validation.py`, `run_sensitivity.py`,
  `generate_paper_figures.py`
- pytest suite: 49 unit + integration tests (was 0)
- README.md, LICENSE (MIT), CITATION.cff, expanded `docs/index.md` and
  `docs/limitations.md`, READMEs for `data/` and `data_raw/`
- CI workflow `.github/workflows/ci.yml` (lint + test on Python 3.11/3.12)
- Logging utility `utils/logging.py` with env-driven level

### Fixed
- `pyproject.toml` - fixed package directory + console_scripts entry, added
  `package-data` so `config/regional/*.toml` ships with the wheel
- `phenology/crop.py` - fixed crop name from "soybean" to "soybeans" and
  JSON key path for crop data loading
- `validation/sensitivity.py` - fatal syntax error (broken `from` import)
  that prevented `import bwb.validation`
- `validation/metrics.py` - replaced 5-function stub with full deterministic
  + probabilistic metric suite (KGE, NSE, RMSE, MAE, bias, pbias, CRPS,
  PIT, alpha-reliability, coverage, interval score)
- `validation/pipeline.py` - wired `run_single_validation` to actually load
  data, fit PyMC, and compute metrics; switched probabilistic metrics to
  use the **posterior predictive** distribution instead of the latent
  theta posterior, fixing coverage_90 from 0.19 -> 0.90
- `models/water_balance.py` - widened `sigma_obs_max` (0.05 -> 0.10) and
  relaxed `sigma_theta_s` (0.03 -> 0.04) and `sigma_theta_init`
  (0.04 -> 0.06) based on Sobol' sensitivity analysis. KGE 0.878 -> 0.880,
  NSE 0.972 -> 0.972, coverage_90 0.19 -> 0.90, pit_alpha 0.62 -> 0.92.
- Validation pipeline output report now uses ASCII (Windows console
  compatibility)

### Changed
- All package modules now ship as proper packages with package-data
- Switched `joblib`-based parallelism to sequential by default on Windows
  (PyMC already parallelises chains internally; loky + multiprocess
  spawn caused pickling errors on win32)

### Removed
- Empty placeholder files in `src/bwb/config/`, `src/bwb/data/`, `src/bwb/utils/`

## [0.1.0] - 2026-04-30

### Added
- Initial release of INFERENCE Bayesian water balance model
- FAO-56 methodology implementation for soybean irrigation
- Historical climate resampling (1961-2025)
- Sequential Bayesian inference with PyMC
- 10-city MATOPIBA region coverage
- Output files: irrigation recommendations, metrics, visualizations

### Known Issues
- Model currently over-predicts irrigation frequency
- KGE and NSE metrics below acceptable thresholds
- Validation against Xavier reanalysis data in progress