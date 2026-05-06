# `data/` -- Curated input data

Curated, version-controlled inputs for the bwb framework. Anything that
should travel with the repository so the validation pipeline can be
reproduced offline lives here.

## Layout

```
data/
+-- Balsas_MA.csv                    Daily Balsas reference series 1961-2025
+-- crops/
|   +-- fao56_table12_kc.json        FAO-56 crop coefficient library
+-- extracted_csv/
|   +-- merged_by_city/              10 MATOPIBA cities, daily climate
|   |   +-- Balsas.csv               date, Rs, u2, Tmax, Tmin, RH, pr, ETo
|   |   +-- Barreiras.csv
|   |   +-- ...
|   +-- by_variable/                 Same data pivoted (city x variable)
+-- soils/
|   +-- soilgrids.py                 Stub for SoilGrids API integration (TODO)
+-- validation/                      External validation datasets (currently empty)
+-- shapefile_matopiba/              Region shapefiles (deleted in HEAD,
                                     restore via `git checkout`)
```

## Conventions

* Every daily series carries a `date` column of dtype `datetime64[ns]`.
* Units: `pr` and `ETo` in mm/day, temperatures in degrees Celsius,
  `u2` in m/s, `Rs` in MJ/m^2/day, `RH` in percent.
* The 10 MATOPIBA cities are: Baixa_Grande_do_Ribeiro, Balsas, Barreiras,
  Bom_Jesus, Campos_Lindos, Correntina, Formosa_do_Rio_Preto,
  Luis_Eduardo_Magalhaes, Tasso_Fragoso, Urucui.

## How to regenerate

The merged-by-city CSVs are produced from the Xavier reanalysis NetCDFs
in `data_raw/xavier/` via:

```bash
python scripts/extract_xavier_matopiba.py
```

The FAO-56 crop library was authored manually from Allen et al. (1998)
Table 12 and is not regenerated automatically.
