"""
Extração diária de variáveis Xavier (BR-DWGD) para os 10 polos MATOPIBA.

Compatível com a estrutura de pastas:
    data/dados_xavier/
        ├── ETo/ETo_*.nc
        ├── pr/pr_*.nc
        ├── RH/RH_*.nc
        ├── Rs/Rs_*.nc
        ├── Tmax/Tmax_*.nc
        ├── Tmin/Tmin_*.nc
        └── u2/u2_*.nc

Pode ser executado de forma incremental: variáveis ainda em download são
detectadas automaticamente e ignoradas sem erro. Re-execute conforme novos
arquivos ficarem disponíveis — apenas as variáveis novas serão processadas
e o merge por cidade é regenerado a cada execução.

Saídas:
    data/extracted_csv/
        ├── by_variable/{cidade}_{var}.csv   # uma série por cidade × variável
        └── merged_by_city/{cidade}.csv      # todas as vars disponíveis juntas

Uso:
    python extract_xavier_matopiba.py
"""

import time
from pathlib import Path

import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO — ajuste apenas estes paths para o seu ambiente
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/dados_xavier")
OUTPUT_DIR = Path("data/extracted_csv")

DATE_START = "1961-01-01"
DATE_END = "2025-12-31"

ALL_VARS = ["Rs", "u2", "Tmax", "Tmin", "RH", "pr", "ETo"]

CITIES = [
    {"nome": "Baixa_Grande_do_Ribeiro", "lat": -7.85,  "lon": -45.24},
    {"nome": "Balsas",                  "lat": -7.53,  "lon": -46.04},
    {"nome": "Barreiras",               "lat": -12.14, "lon": -45.00},
    {"nome": "Bom_Jesus",               "lat": -9.07,  "lon": -44.36},
    {"nome": "Campos_Lindos",           "lat": -7.97,  "lon": -46.80},
    {"nome": "Correntina",              "lat": -13.33, "lon": -44.62},
    {"nome": "Formosa_do_Rio_Preto",    "lat": -11.05, "lon": -45.19},
    {"nome": "Luis_Eduardo_Magalhaes",  "lat": -12.08, "lon": -45.80},
    {"nome": "Tasso_Fragoso",           "lat": -8.48,  "lon": -45.74},
    {"nome": "Urucui",                  "lat": -7.23,  "lon": -44.56},
]

# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def detect_available_vars(data_dir: Path) -> dict:
    """Retorna {var_name: [arquivos .nc ordenados]} apenas para vars com dados."""
    available = {}
    for var in ALL_VARS:
        var_dir = data_dir / var
        if not var_dir.exists():
            continue
        files = sorted(var_dir.glob(f"{var}_*.nc"))
        if files:
            available[var] = files
    return available


def extract_variable(var_name: str, files: list, cities: list) -> tuple:
    """Abre cada NetCDF da variável separadamente, faz a seleção espacial dos
    10 pontos antes de materializar (evita estouro de memória com a grade
    completa), e concatena os resultados ao longo do tempo."""
    print(f"\n[{var_name}] processando {len(files)} arquivo(s)...")
    t0 = time.time()

    lats = [c["lat"] for c in cities]
    lons = [c["lon"] for c in cities]

    parts = []
    for fpath in files:
        # Chunks pequenos no tempo + seleção espacial antes de .compute()
        # mantém apenas (chunk_dias × 10 pontos) na memória, nunca a grade cheia.
        with xr.open_dataset(fpath, chunks={"time": 365}) as ds:
            da = ds[var_name].sel(time=slice(DATE_START, DATE_END))
            da = da.sel(
                longitude=xr.DataArray(lons, dims="city"),
                latitude=xr.DataArray(lats, dims="city"),
                method="nearest",
            )
            da_loaded = da.compute()  # materializa apenas (tempo × 10 cidades)
            parts.append(da_loaded)
            print(f"  {fpath.name}: shape={da_loaded.shape}  "
                  f"{pd.to_datetime(da_loaded.time.values).min().date()} → "
                  f"{pd.to_datetime(da_loaded.time.values).max().date()}")

    if not parts:
        raise RuntimeError(f"Nenhum dado extraído para {var_name}")

    da_full = xr.concat(parts, dim="time").sortby("time")
    values = da_full.values            # shape: (tempo, cidade)
    times = pd.to_datetime(da_full.time.values)

    elapsed = time.time() - t0
    print(f"[{var_name}] consolidado shape={values.shape}  "
          f"período={times.min().date()} → {times.max().date()}  "
          f"({elapsed:.1f}s)")
    return values, times


def save_per_variable(values, times, var_name: str, cities: list, out_dir: Path):
    """Salva um CSV por cidade × variável (persistente, permite extração incremental)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, city in enumerate(cities):
        df = pd.DataFrame({var_name: values[:, i]}, index=times)
        df.index.name = "date"
        out_file = out_dir / f"{city['nome']}_{var_name}.csv"
        df.to_csv(out_file, float_format="%.2f")


def merge_by_city(cities: list, in_dir: Path, out_dir: Path):
    """Concatena todas as variáveis disponíveis por cidade num único CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Merge por cidade ---")
    for city in cities:
        dfs = []
        vars_found = []
        for var in ALL_VARS:
            f = in_dir / f"{city['nome']}_{var}.csv"
            if f.exists():
                dfs.append(pd.read_csv(f, index_col="date", parse_dates=True))
                vars_found.append(var)
        if not dfs:
            print(f"  {city['nome']:32s} -- sem dados ainda")
            continue
        merged = pd.concat(dfs, axis=1)
        merged = merged[[v for v in ALL_VARS if v in vars_found]]  # ordem canônica
        out_file = out_dir / f"{city['nome']}.csv"
        merged.to_csv(out_file, float_format="%.2f")
        n_nan = merged.isna().sum().sum()
        nan_str = f"  ({n_nan} NaN)" if n_nan > 0 else ""
        print(f"  {city['nome']:32s} -> {len(vars_found)} vars × {len(merged)} dias{nan_str}")


def sanity_check(merged_dir: Path):
    """Verifica integridade básica dos CSVs gerados."""
    print("\n--- Sanity check ---")
    csvs = sorted(merged_dir.glob("*.csv"))
    if not csvs:
        print("  nenhum CSV gerado.")
        return
    for f in csvs:
        df = pd.read_csv(f, index_col="date", parse_dates=True)
        n_days = len(df)
        n_vars = len(df.columns)
        date_min = df.index.min().date()
        date_max = df.index.max().date()
        print(f"  {f.stem:32s}  {n_days:6d} dias  {n_vars} vars  {date_min} → {date_max}")


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def main():
    available = detect_available_vars(DATA_DIR)
    missing = sorted(set(ALL_VARS) - set(available.keys()))

    print("=" * 78)
    print(" Extração Xavier BR-DWGD → 10 polos MATOPIBA ")
    print("=" * 78)
    print(f"Período solicitado : {DATE_START} → {DATE_END}")
    print(f"Cidades            : {len(CITIES)}")
    print(f"Variáveis com dados: {list(available.keys())}")
    print(f"Variáveis pendentes: {missing if missing else 'nenhuma'}")
    print("=" * 78)

    if not available:
        print("\nNenhuma variável encontrada em", DATA_DIR.resolve())
        print("Verifique o path em DATA_DIR no topo do script.")
        return

    var_dir = OUTPUT_DIR / "by_variable"
    merged_dir = OUTPUT_DIR / "merged_by_city"

    for var_name, files in available.items():
        values, times = extract_variable(var_name, files, CITIES)
        save_per_variable(values, times, var_name, CITIES, var_dir)

    merge_by_city(CITIES, var_dir, merged_dir)
    sanity_check(merged_dir)

    print("\nConcluído.")
    print(f"Arquivos por variável : {var_dir.resolve()}")
    print(f"Arquivos por cidade   : {merged_dir.resolve()}")
    if missing:
        print(f"\nQuando os downloads de {missing} terminarem, basta re-executar este script.")
        print("As variáveis já extraídas não serão reprocessadas; apenas as novas.")


if __name__ == "__main__":
    main()