"""
download_oni.py
===============

Baixa o Oceanic Niño Index (ONI) oficial da NOAA/CPC, parseia o formato ASCII
tabular, recorta para o período 1961-2025, e salva em Parquet + CSV.

Fonte:
    https://psl.noaa.gov/data/correlation/oni.data
    (mirror NOAA/PSL do arquivo gerado pela CPC, baseado em ERSST.v5)

Saídas:
    data/oceanic_raw/oni_raw.txt       — cópia bruta do arquivo baixado
    data/oceanic_processed/oni.parquet — série mensal limpa (date, oni)
    data/oceanic_processed/oni.csv     — mesma série em CSV (legível humano)

Uso:
    python download_oni.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

URL = "https://psl.noaa.gov/data/correlation/oni.data"

RAW_DIR = Path("data/oceanic_raw")
OUT_DIR = Path("data/oceanic_processed")
RAW_FILE = RAW_DIR / "oni_raw.txt"
PARQUET_FILE = OUT_DIR / "oni.parquet"
CSV_FILE = OUT_DIR / "oni.csv"

DATE_START = "1961-01-01"
DATE_END = "2025-12-31"

MISSING_VALUE = -99.9

# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------


def download(url: str, dest: Path) -> Path:
    """Baixa o arquivo bruto. Mantém cópia local em formato original."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando ONI de {url} ...")
    try:
        with urlopen(url, timeout=30) as resp:
            content = resp.read()
    except URLError as e:
        raise RuntimeError(f"Falha no download de {url}: {e}")

    dest.write_bytes(content)
    n_kb = len(content) / 1024
    print(f"  Arquivo bruto salvo em {dest} ({n_kb:.1f} KB)")
    return dest


def parse_oni(raw_file: Path) -> pd.DataFrame:
    """Parseia o arquivo ASCII tabular do ONI para DataFrame long format."""
    print(f"Parseando {raw_file.name} ...")

    # Estrutura do arquivo:
    #   linha 1: " <ano_inicio>  <ano_fim>"
    #   linhas 2..N: " <ano>  <jan>  <fev>  ...  <dez>"
    #   últimas ~8 linhas: rodapé com metadata
    #
    # Como o número exato de linhas de rodapé pode variar com o tempo,
    # usamos uma estratégia robusta: ler linha por linha, ignorando
    # qualquer linha cuja primeira "palavra" não seja um ano numérico.

    rows = []
    with open(raw_file, encoding="utf-8") as f:
        first = f.readline().split()
        year_start = int(first[0])
        year_end = int(first[1])

        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                # linha de rodapé (texto, não número)
                continue
            if year < year_start or year > year_end:
                continue
            if len(parts) < 13:
                continue
            try:
                values = [float(x) for x in parts[1:13]]
            except ValueError:
                continue
            rows.append([year] + values)

    print(f"  {len(rows)} anos de dados encontrados ({year_start}-{year_end})")

    df = pd.DataFrame(rows, columns=["year"] + list(range(1, 13)))

    # Pivot para long format: (year, month, oni)
    df = df.melt(id_vars="year", var_name="month", value_name="oni")
    df["month"] = df["month"].astype(int)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))

    # Remove missing values (-99.9 indica mês sem dado, tipicamente futuro)
    df = df[df["oni"] != MISSING_VALUE].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "year", "month", "oni"]]

    return df


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Recorta a série para o intervalo desejado (inclusivo)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


def save_outputs(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    """Persiste em CSV (sempre) e Parquet (se pyarrow/fastparquet disponíveis)."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV primeiro — não depende de bibliotecas externas
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"  CSV     : {csv_path}")

    # Parquet — requer pyarrow ou fastparquet
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"  Parquet : {parquet_path}")
    except ImportError:
        print(f"  Parquet : PULADO (instale com: pip install pyarrow)")
    except Exception as e:
        print(f"  Parquet : ERRO ao salvar ({type(e).__name__}: {e})")


def summary(df: pd.DataFrame) -> None:
    """Imprime resumo de qualidade dos dados."""
    print("\n--- Resumo da série ONI ---")
    print(f"  Período         : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total de meses  : {len(df)}")
    print(f"  Anos cobertos   : {df['year'].nunique()}")
    print(f"  ONI mínimo      : {df['oni'].min():.2f}  ({df.loc[df['oni'].idxmin(), 'date'].date()})")
    print(f"  ONI máximo      : {df['oni'].max():.2f}  ({df.loc[df['oni'].idxmax(), 'date'].date()})")
    print(f"  ONI médio       : {df['oni'].mean():+.3f}")
    print(f"  Desvio-padrão   : {df['oni'].std():.3f}")

    n_el_nino = ((df["oni"] >= 0.5)).sum()
    n_la_nina = ((df["oni"] <= -0.5)).sum()
    n_neutro = len(df) - n_el_nino - n_la_nina
    print(f"  Meses El Niño   : {n_el_nino:4d}  (ONI ≥ +0.5)")
    print(f"  Meses La Niña   : {n_la_nina:4d}  (ONI ≤ -0.5)")
    print(f"  Meses neutros   : {n_neutro:4d}")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print(" Download e parsing do ONI (Oceanic Niño Index)")
    print("=" * 70)

    raw_path = download(URL, RAW_FILE)
    df_full = parse_oni(raw_path)
    df = filter_period(df_full, DATE_START, DATE_END)

    print(f"\nApós recorte para {DATE_START} → {DATE_END}: {len(df)} meses")

    save_outputs(df, PARQUET_FILE, CSV_FILE)
    summary(df)

    print("\nConcluído.")


if __name__ == "__main__":
    main()