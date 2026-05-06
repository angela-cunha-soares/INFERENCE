---
description: "Use when: working on the INFERENCE water balance model for soybean irrigation in Balsas (MA), Brazil"
---

# INFERENCE - Modelo de Balanço Hídrico para Soja

## Project Overview

This is a Bayesian sequential inference model for soybean water balance analysis in Balsas, MA, Brazil. The model uses FAO-56 methodology, historical climate resampling, and PyMC for Bayesian inference.

## Key Files

| File | Purpose |
|------|---------|
| [main.py](main.py) | Main entry point - runs the complete water balance model |
| [gera_curva_retencao.py](gera_curva_retencao.py) | Generates soil water retention curves (Van Genuchten) |
| [data/Balsas_MA.csv](data/Balsas_MA.csv) | Historical climate data (1961-present): ETo (evapotranspiration) and precipitation |
| [output/](output/) | Model outputs: CSV files with irrigation recommendations and metrics |

## Running the Model

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run main model
python main.py
```

## Important Conventions

- **Crop cycle**: 90 days, planting December 1st
- **Root system depth**: 60 cm
- **Available water capacity (AWC)**: 120 mm
- **Kc values by phase**: Initial (0.40) → Development (0.80) → Intermediate (1.15) → Final (0.80) → Harvest (0.50)
- **Target cycles**: 2020-2024 (configurable in `CICLOS_ALVO`)
- **Samples**: 2000 samples, 500 simulations

## Dependencies

- pandas, numpy
- pymc (Bayesian inference)
- arviz (MCMC diagnostics)
- matplotlib, seaborn (visualization)
- scipy (statistical functions)

## Data Format

Input CSV (`data/Balsas_MA.csv`):
- `Data`: Date (YYYY-MM-DD)
- `ETo`: Reference evapotranspiration (mm/day)
- `pr`: Precipitation (mm)

## Output Files

- `deterministico_YYYY_YYYY.csv`: Deterministic results
- `previsao_YYYY_YYYY.csv`: Forecast results
- `laminas_recomendadas.csv`: Recommended irrigation depths
- `metricas_probabilisticas.csv`: Probabilistic metrics
- `metricas_sequenciais.csv`: Sequential metrics