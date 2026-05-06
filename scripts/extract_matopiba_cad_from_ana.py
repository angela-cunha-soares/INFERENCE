"""Extract per-municipality CAD for MATOPIBA cities from ANA/SNIRH dataset.

Source
------
Sistema Nacional de Informações sobre Recursos Hídricos (SNIRH/ANA),
"Capacidade de Água Disponível por Município (Brasil)":
https://metadados.snirh.gov.br/geonetwork/srv/api/records/28fe4baa-66f3-4f6b-b0d2-890abf5910c4

The SNIRH file is tab-separated, Latin-1 encoded, with comma-decimal numbers and
columns: ``Codigo<TAB>Municipio<TAB>UF<TAB>CAD``. CAD is in m^3/m^3 (volumetric
fraction of available water).

Computation
-----------
For an effective root depth ``Z_r`` (cm), the available water capacity in mm is::

    AWC_mm = CAD (m^3/m^3) * Z_r (cm) * 10 (mm/cm)

The 90-day soybean cycle uses Z_r = 60 cm (matopiba.toml). The SNIRH metadata
should be consulted to confirm what depth ANA integrated when computing CAD;
this script reports both 60 cm and 100 cm to bracket the FAO-56 range.

Usage
-----
::

    # 1) save the raw SNIRH text file to data_raw/soils/cad_ana.txt (Latin-1)
    # 2) run:
    python scripts/extract_matopiba_cad_from_ana.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data_raw" / "soils" / "cad_ana.txt"
OUT_DIR = ROOT / "data" / "soils"
OUT_CSV = OUT_DIR / "matopiba_cad.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 MATOPIBA municipalities used in the validation grid.
# IBGE codes verified against matopiba.toml [cities].
MATOPIBA = {
    2201150: ("Baixa_Grande_do_Ribeiro", "PI"),
    2101400: ("Balsas",                  "MA"),
    2903201: ("Barreiras",               "BA"),
    2201903: ("Bom_Jesus",               "PI"),
    1703842: ("Campos_Lindos",           "TO"),
    2909307: ("Correntina",              "BA"),
    2911105: ("Formosa_do_Rio_Preto",    "BA"),
    2919553: ("Luis_Eduardo_Magalhaes",  "BA"),
    2112001: ("Tasso_Fragoso",           "MA"),
    2211209: ("Urucui",                  "PI"),
}

ROOT_DEPTH_CM_DEFAULT = 60   # matopiba.toml [crop].root_depth_cm
ROOT_DEPTH_CM_FAO56_MAX = 100  # FAO-56 soybean mature root zone


def parse_snirh_txt(path: Path) -> dict[int, tuple[str, str, float]]:
    """Yield (ibge_code, municipio, uf, cad) for every row.

    Returns a dict keyed by IBGE code containing only MATOPIBA cities.
    """
    out: dict[int, tuple[str, str, float]] = {}
    with path.open(encoding="latin-1") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line or "Cdigo" in line or "Codigo" in line or "C\xf3digo" in line:
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 4:
                continue
            try:
                code = int(parts[0])
            except ValueError:
                continue
            if code not in MATOPIBA:
                continue
            municipio = parts[1]
            uf = parts[2]
            try:
                cad = float(parts[3].replace(",", "."))
            except ValueError:
                print(f"  [warn] line {line_no}: cannot parse CAD '{parts[3]}'",
                      file=sys.stderr)
                continue
            out[code] = (municipio, uf, cad)
    return out


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: missing {RAW_PATH.relative_to(ROOT)}\n"
              f"Save the SNIRH 'cad_ana.txt' file there first.",
              file=sys.stderr)
        sys.exit(1)

    parsed = parse_snirh_txt(RAW_PATH)
    missing = sorted(set(MATOPIBA) - set(parsed))
    if missing:
        print(f"WARNING: missing IBGE codes in source: {missing}", file=sys.stderr)

    rows = []
    print("=" * 92)
    print(f" MATOPIBA CAD from SNIRH/ANA  (root depths: {ROOT_DEPTH_CM_DEFAULT} cm "
          f"and {ROOT_DEPTH_CM_FAO56_MAX} cm)")
    print("=" * 92)
    print(f"{'City':<28} {'UF':<3} {'CAD m3/m3':>10}  "
          f"{'AWC60 mm':>9}  {'AWC100 mm':>10}")
    print("-" * 92)
    for code, (city_pep8, uf) in MATOPIBA.items():
        if code not in parsed:
            continue
        municipio_raw, uf_raw, cad = parsed[code]
        awc60 = cad * ROOT_DEPTH_CM_DEFAULT * 10
        awc100 = cad * ROOT_DEPTH_CM_FAO56_MAX * 10
        rows.append({
            "ibge_code":           code,
            "city":                city_pep8,
            "municipio_snirh":     municipio_raw,
            "uf":                  uf_raw,
            "cad_m3_m3":           round(cad, 6),
            "root_depth_cm":       ROOT_DEPTH_CM_DEFAULT,
            "awc_mm_60cm":         round(awc60, 2),
            "awc_mm_100cm":        round(awc100, 2),
            "source":              "SNIRH/ANA dataset 28fe4baa",
        })
        print(f"{city_pep8:<28} {uf_raw:<3} {cad:>10.4f}  "
              f"{awc60:>9.1f}  {awc100:>10.1f}")
    print("-" * 92)
    if rows:
        cad_vals = [r["cad_m3_m3"] for r in rows]
        awc60_vals = [r["awc_mm_60cm"] for r in rows]
        print(f"{'mean':<28} {'':<3} {sum(cad_vals)/len(cad_vals):>10.4f}  "
              f"{sum(awc60_vals)/len(awc60_vals):>9.1f}")
        print(f"{'min/max CAD':<28} {'':<3} "
              f"{min(cad_vals):.4f} / {max(cad_vals):.4f}")
        print(f"{'matopiba.toml current':<28} {'':<3} {'':>10}  "
              f"{120.0:>9.1f}  (uniform, hardcoded)")

    if rows:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  -> {OUT_CSV.relative_to(ROOT)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
