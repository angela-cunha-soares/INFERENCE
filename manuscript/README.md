# Manuscript -- Computers and Electronics in Agriculture submission

Source files for the manuscript

> *A reproducible Bayesian water-balance framework for risk-aware soybean
> irrigation: open-source software and validation across the Brazilian
> MATOPIBA agricultural frontier*

targeted at **Computers and Electronics in Agriculture** (Elsevier).

## Files

* `manuscript.tex` -- the main LaTeX source (Elsevier `elsarticle.cls`,
  review style, double-spaced single-column with line numbers).
* `references.bib` -- full bibliography in BibTeX format
  (Elsevier numeric style).
* `../figures/paper/` -- figures referenced by the manuscript.

## Compiling

```bash
cd manuscript
pdflatex manuscript
bibtex   manuscript
pdflatex manuscript
pdflatex manuscript
```

Or, if you prefer `latexmk`:

```bash
latexmk -pdf manuscript.tex
```

The Elsevier `elsarticle` class is included in any reasonable TeX Live
or MikTeX distribution -- no additional packages need to be installed.

## Placeholders to fill before submission

Search the source for `[...]` to find every spot that requires a
human decision:

1. `[Author surname]`, `[corresponding author email]`
2. `[Graduate Programme]`, `[Street, number]`, `[City]`, `[ZIP]`
3. `[user]` in the GitHub URL (currently `https://github.com/[user]/INFERENCE`)
4. `[XXXX]` in the Zenodo DOI placeholder
5. `[funding agencies, grant numbers]` in the Acknowledgements
6. `[Author~1]` in the CRediT statement

## Numbers reported

Every numerical value in the manuscript is reproducible from the open
artifacts in this repository. Two helper scripts bundled with the
project verify both consistency and freshness:

```bash
python scripts/audit_labels.py   # \label vs \ref completeness
python scripts/audit_numbers.py  # every numerical claim vs CSVs/JSONs
```

| Section | Source |
|---|---|
| Posterior-recovery (Table `tab:posterior_recovery`, `tab:per_depth`) | `output/paper_tables/table_global_metrics.csv`, `output/paper_tables/table_per_depth.csv` (reproduce with `bwb posterior-recovery`) |
| MCMC diagnostics (Table `tab:mcmc-diagnostics`) | `output/paper_tables/table_mcmc_diagnostics.csv` (reproduce with `scripts/generate_mcmc_diagnostics.py`) |
| Sobol' sensitivity (Table `tab:sobol`) | `output/sensitivity/sobol_Balsas_2023.json` (reproduce with `bwb sensitivity --city Balsas --cycle 2023 --n 1000`) |
| Climatological forecast (Tables `tab:forecast_seq`, `tab:forecast_by_city`, `tab:forecast_by_cycle`, `tab:sequential_all_cities`) | `output/forecast_sequential/sequential_summary_all_cities.csv` (reproduce with `bwb forecast-sequential --n-sim 500`) |
| Climatological-forecast figures (`fig:forecast_calibration`, `fig:forecast_intervals_by_cycle`, `fig:forecast_trajectory_balsas`) | `python scripts/generate_forecast_seq_figures.py` (reads `output/forecast_sequential/`) |
| Operational rolling forecast (Table `tab:backtest-rolling`) | `output/backtest_rolling/backtest_rolling_h5d.csv` (reproduce with `bwb backtest-rolling`) |
| DAG / DBN (Tables `tab:dag-relevance`, `tab:dbn-enso`) | `output/paper_tables/table_dag_relevance.csv`, `output/paper_tables/table_dbn_enso.csv` |
| Calibration history (Table `tab:calibration_history`) | `CHANGELOG.md` and version-controlled commits |
| INMET fused-product validation (Appendix A, Table `tab:inmet-validation`) | `output/inmet_validation/per_station_summary.csv`, `output/inmet_validation/pooled_summary.csv` |
| ETo alternatives (Table `tab:eto-alternatives`) | `output/paper_tables/table_eto_alternatives.csv` |

## Status (snapshot)

* Methodology: aligned with the implemented codebase.
* Results: real numbers from 4{,}450 runs (150 posterior-recovery + 50
  climatological forecast + 4{,}250 rolling 5-day forecasts).
* Discussion: drafted with explicit limitations and roadmap.
* Conclusion: written, ends with risk-aware decision-support anchor.
* American English throughout (`scripts/audit_labels.py` and
  `scripts/audit_numbers.py` confirm zero orphan labels and zero stale
  numbers as of the last commit).
* References: 50+ entries in `references.bib`.
* Appendices: A (INMET fused-product validation), B (DAG climate
  structure learning), C (sequential forecast detailed results), D
  (daily SW trajectory atlas at Balsas).

## Items NOT yet implemented but mentioned only as future work

The text is honest about these. They are *not* claimed as results:

* Bayesian-network imputation of missing meteorological variables
  (Discussion: future work hook).
* Hierarchical partial pooling across cities
  (Discussion: future work).
* Daily rolling forecast with real GEFS reforecast
  (Discussion: future work; synthetic ensemble already in the
  software via `bwb forecast --members 31`).
* External validation against in-situ or satellite soil-moisture
  products such as GLEAM SMroot or ESA CCI SM
  (Discussion: future work).
* Operational forecast for cycle 2025/26 with INMET January-February 2026
  data (the framework is ready; awaiting raw data).

## Pre-submission checklist

* [ ] Fill the six author / institutional placeholders.
* [ ] Replace `[user]` with the GitHub user.
* [ ] Replace `[XXXX]` with the Zenodo DOI obtained at submission time.
* [ ] Update the Acknowledgements with funding details.
* [ ] Compile, proofread, run the spell checker.
* [ ] Generate the high-resolution PDF/EPS versions of each figure.
* [ ] Submit the cover letter (separate file, see Elsevier guidelines).
