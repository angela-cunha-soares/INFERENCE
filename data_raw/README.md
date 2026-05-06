# `data_raw/` -- Raw third-party downloads

Original (un-tampered) sources. Files here are inputs to the processing
scripts that populate `data_processed/` and `data/extracted_csv/`. They are
**not** edited in place -- regenerate them with the download/extraction
scripts when the upstream sources update.

## Layout

```
data_raw/
+-- xavier/                  Xavier reanalysis NetCDF (per variable, 1961-2025)
|   +-- pr/   pr_19610101_19801231.nc, pr_19810101_20001231.nc, pr_20010101_20251231.nc
|   +-- ETo/  ...
|   +-- Tmax/, Tmin/, RH/, Rs/, u2/
+-- oceanic/                 Raw oceanic indices (NOAA / PSL)
|   +-- oni_raw.txt          Oceanic Nino Index, 1950-present (download_oni.py)
|   +-- mei_raw.txt          Multivariate ENSO Index v2, 1979-present (download_mei.py)
|   +-- amm_raw.txt          (TODO) Atlantic Meridional Mode
|   +-- amo_raw.txt          (TODO) Atlantic Multidecadal Oscillation
|   +-- pdo_raw.csv          (TODO) Pacific Decadal Oscillation
|   +-- mjo_raw.txt          (TODO) RMM1/RMM2 daily indices
|   +-- iod_raw.txt          (TODO) Indian Ocean Dipole
+-- soil_moisture/           Reserved for an external soil-moisture product
                              (e.g. GLEAM SMroot, ESA CCI SM, GLDAS-2);
                              Xavier itself does NOT publish soil moisture.
```

## How to regenerate

```bash
# ENSO indices (already shipped)
python scripts/download_oni.py
python scripts/download_mei.py

# Xavier (each variable / period must be downloaded separately from
# https://www.cpc.ncep.noaa.gov/products/xavier/ or via OPeNDAP). The
# extraction step assumes the directory layout shown above.
python scripts/extract_xavier_matopiba.py

# Consolidate oceanic indices into a single Parquet (handles missing files)
python scripts/oceanic_indices_consolidator.py
```

## Notes

* Raw files are large (the 7-variable Xavier set is ~21 GB) and not all are
  tracked by git -- consult the project owner if you need a snapshot.
* The consolidator gracefully skips indices that aren't present, so a
  partial download of `oceanic/` is still usable.
