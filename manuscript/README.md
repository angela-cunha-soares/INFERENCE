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
artefacts in this repository:

| Section | Source |
|---|---|
| Posterior-recovery (Table 2) | `output/validation/validation_*.csv` reproduced with `bwb posterior-recovery` |
| Sobol' sensitivity (Table 3) | `output/sensitivity/sobol_Balsas_2023.json` reproduced with `bwb sensitivity --city Balsas --cycle 2023 --n 1000` |
| Climatological forecast (Tables 4-6) | `output/forecast_sequential/sequential_summary_all_cities.csv` reproduced with `bwb forecast-sequential --n-sim 500` |
| Calibration history (Table 7) | `CHANGELOG.md` and version-controlled commits |

## Status (snapshot)

* Methodology section: aligned with the implemented codebase.
* Results section: filled with real numbers from the 200 validation runs
  (150 posterior-recovery + 50 climatological forecast).
* Discussion: drafted with explicit limitations and roadmap.
* Conclusion: written.
* References: 26 entries, complete for the current text.

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
