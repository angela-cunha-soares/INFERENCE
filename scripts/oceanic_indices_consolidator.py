"""
oceanic_indices_consolidator.py
================================

Consolida os 7 índices oceânicos baixados em formatos heterogêneos num
único Parquet alinhado mensalmente para o período 1961-2025.

Inputs (esperados em data/oceanic_raw/):
    oni_raw.txt       (NOAA/PSL, ONI 1950-presente)
    mei_raw.txt       (NOAA/PSL, MEI.v2 1979-presente)
    amm_raw.txt       (NOAA/PSL, AMM SST 1948-presente)
    amo_raw.txt       (NOAA/PSL AMO 1856+ ou 1948+)
    pdo_raw.csv       (NCEI/NOAA, 1854+)
    mjo_raw.txt       (BOM, RMM diário 1974+)
    iod_raw.txt       (NOAA/PSL, DMI 1870+)

Output (em data_processed/oceanic_processed/):
    oceanic_indices.parquet   monthly (date, oni, mei, amm, amo, pdo, mjo_amp, iod)
    oceanic_indices.csv       mesma série em formato legível

Para os índices não baixados ainda, o consolidator preenche com NaN sem
falhar — útil para iterar enquanto downloads completam.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / "data" / "oceanic_raw").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return here


PROJECT_ROOT = _find_project_root()
RAW_DIR = PROJECT_ROOT / "data" / "oceanic_raw"
OUT_DIR = PROJECT_ROOT / "data_processed" / "oceanic_processed"

DATE_START = "1961-01-01"
DATE_END = "2025-12-31"

# ---------------------------------------------------------------------------
# Parsers — formato NOAA/PSL "ano + 12 meses" (usado para ONI, AMO, AMM, IOD)
# ---------------------------------------------------------------------------


def parse_psl_yearly_table(path: Path, missing_values=(-99.9, -99.99, -999.0)) -> pd.Series:
    """Parser robusto para tabelas NOAA/PSL no formato:
        <ano_inicio>  <ano_fim>
        1950  -1.53  -1.34  -1.16 ...  (12 valores)
        ...
        rodapé textual
    Retorna Series indexada por data (mês inicial).
    """
    if not path.exists():
        return pd.Series(dtype=float)

    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().split()
        try:
            year_start = int(first[0])
            year_end = int(first[1])
        except (IndexError, ValueError):
            return pd.Series(dtype=float)

        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if year < year_start or year > year_end:
                continue
            if len(parts) < 13:
                continue
            try:
                vals = [float(x) for x in parts[1:13]]
            except ValueError:
                continue
            rows.append([year] + vals)

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["year"] + list(range(1, 13)))
    df = df.melt(id_vars="year", var_name="month", value_name="value")
    df["month"] = df["month"].astype(int)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    # Remove missing values
    for mv in missing_values:
        df = df[df["value"] != mv]
    df = df.dropna(subset=["value"]).sort_values("date")
    s = df.set_index("date")["value"].astype(float)
    s.name = None
    return s


# ---------------------------------------------------------------------------
# MEI.v2 — bimestral, atribuído ao 2° mês da janela
# ---------------------------------------------------------------------------


def parse_mei(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().split()
        try:
            year_start = int(first[0])
            year_end = int(first[1])
        except (IndexError, ValueError):
            return pd.Series(dtype=float)
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if year < year_start or year > year_end:
                continue
            if len(parts) < 13:
                continue
            try:
                vals = [float(x) for x in parts[1:13]]
            except ValueError:
                continue
            rows.append([year] + vals)
    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["year"] + list(range(1, 13)))
    df = df.melt(id_vars="year", var_name="month", value_name="value")
    df["month"] = df["month"].astype(int)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df = df[df["value"] != -999.00].dropna(subset=["value"])
    df = df.sort_values("date")
    s = df.set_index("date")["value"].astype(float)
    s.name = None
    return s


# ---------------------------------------------------------------------------
# PDO — formato CSV NCEI (data, valor)
# ---------------------------------------------------------------------------


def parse_pdo(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    try:
        # Formato típico NCEI: skiprows variável, colunas Date, Value
        df = pd.read_csv(path, comment="#")
    except Exception:
        return pd.Series(dtype=float)
    # Normaliza nome das colunas
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = next((cols_lower[c] for c in ("date", "month", "yyyy-mm")
                     if c in cols_lower), df.columns[0])
    val_col = next((cols_lower[c] for c in ("value", "pdo", "index", "anomaly")
                    if c in cols_lower), df.columns[-1])
    try:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        return pd.Series(dtype=float)
    df = df.dropna(subset=["date"])
    # Força dia 1
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["value"] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    s = df.set_index("date")["value"].astype(float)
    s.name = None
    return s


# ---------------------------------------------------------------------------
# MJO RMM — formato BOM, diário, agregado para amplitude mensal
# ---------------------------------------------------------------------------


def parse_mjo(path: Path) -> pd.Series:
    """RMM BOM: agrega para amplitude mensal média.

    Formato: header textual + linhas "year month day RMM1 RMM2 phase amplitude origin"
    """
    if not path.exists():
        return pd.Series(dtype=float)

    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 7:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                amp = float(parts[6])
                if amp == 999 or amp == 1.E+36:
                    continue
            except (ValueError, IndexError):
                continue
            if 1900 < year < 2100 and 1 <= month <= 12:
                rows.append((pd.Timestamp(year=year, month=month, day=day), amp))

    if not rows:
        return pd.Series(dtype=float)

    daily = pd.DataFrame(rows, columns=["date", "amp"]).set_index("date")
    monthly = daily["amp"].resample("MS").mean()
    monthly = monthly.dropna()
    return monthly


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


INDEX_LOADERS = {
    "oni": ("oni_raw.txt",  parse_psl_yearly_table),
    "mei": ("mei_raw.txt",  parse_mei),
    "amm": ("amm_raw.txt",  parse_psl_yearly_table),
    "amo": ("amo_raw.txt",  parse_psl_yearly_table),
    "iod": ("iod_raw.txt",  parse_psl_yearly_table),
    "pdo": ("pdo_raw.csv",  parse_pdo),
    "mjo_amp": ("mjo_raw.txt", parse_mjo),
}


def consolidate() -> pd.DataFrame:
    print("=" * 78)
    print(" Oceanic indices consolidator")
    print("=" * 78)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw dir      : {RAW_DIR}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Period       : {DATE_START} → {DATE_END}")
    print()

    series_dict = {}
    for name, (fname, parser) in INDEX_LOADERS.items():
        path = RAW_DIR / fname
        if path.exists():
            s = parser(path)
            n = len(s)
            if n > 0:
                series_dict[name] = s
                print(f"  {name:8s} : {n:5d} months, "
                      f"{s.index.min().date()} → {s.index.max().date()}")
            else:
                print(f"  {name:8s} : empty (parser returned 0 rows)")
        else:
            print(f"  {name:8s} : MISSING ({path.name})")

    if not series_dict:
        print("\nNenhum índice carregado. Verifique data/oceanic_raw/.")
        return pd.DataFrame()

    # Index mensal completo no período alvo
    full_idx = pd.date_range(DATE_START, DATE_END, freq="MS")
    df = pd.DataFrame(index=full_idx)
    df.index.name = "date"
    for name, s in series_dict.items():
        df[name] = s.reindex(full_idx)

    # Adiciona colunas de coverage
    n_total = len(df)
    print(f"\n  Consolidated frame: {n_total} months × {len(df.columns)} indices")
    for col in df.columns:
        n_valid = df[col].notna().sum()
        coverage = 100 * n_valid / n_total
        print(f"    {col:8s}: {n_valid:4d}/{n_total} ({coverage:.0f}% coverage)")

    return df


def save_outputs(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    # CSV
    csv_path = OUT_DIR / "oceanic_indices.csv"
    df.to_csv(csv_path, float_format="%.3f")
    print(f"\n  CSV      : {csv_path}")

    # Parquet
    parquet_path = OUT_DIR / "oceanic_indices.parquet"
    try:
        df.reset_index().to_parquet(parquet_path, index=False)
        print(f"  Parquet  : {parquet_path}")
    except ImportError:
        print(f"  Parquet  : PULADO (instale pyarrow)")

    # Resumo descritivo (média, std por índice)
    summary_path = OUT_DIR / "oceanic_indices_summary.csv"
    df.describe().to_csv(summary_path, float_format="%.3f")
    print(f"  Summary  : {summary_path}")


def main() -> None:
    df = consolidate()
    save_outputs(df)
    print("\nConcluído.")


if __name__ == "__main__":
    main()