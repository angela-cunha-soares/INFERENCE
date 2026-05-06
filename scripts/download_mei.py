"""
download_mei.py
===============

Baixa o Multivariate ENSO Index version 2 (MEI.v2) da NOAA/PSL, parseia o
formato ASCII bimestral, alinha à frequência mensal (2-month season →
segundo mês da janela), recorta para 1961-2025 e salva em Parquet + CSV.

Fonte:
    https://psl.noaa.gov/enso/mei/data/meiv2.data

Estrutura do arquivo (12 colunas após o ano):
    DJ  JF  FM  MA  AM  MJ  JJ  JA  AS  SO  ON  ND
    │   │   │   │   │   │   │   │   │   │   │   └─ Nov+Dez  → atribuído a Dezembro
    │   │   │   │   │   │   │   │   │   │   └───── Out+Nov  → atribuído a Novembro
    │   │   │   │   │   │   │   │   │   └───────── Set+Out  → atribuído a Outubro
    │   │   │   │   │   │   │   │   └───────────── Ago+Set  → atribuído a Setembro
    ...
    └───── Dez(Y-1)+Jan  → atribuído a Janeiro do ano Y

Período coberto: 1979-presente (anos anteriores a 1979 ficam como NaN no
modelo, comportamento natural — o likelihood do MEI simplesmente não
contribui no log-posterior dos meses sem dado).

Saídas:
    data/oceanic_raw/mei_raw.txt
    data/oceanic_processed/mei.parquet
    data/oceanic_processed/mei.csv

Uso:
    python download_mei.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"

RAW_DIR = Path("data/oceanic_raw")
OUT_DIR = Path("data/oceanic_processed")
RAW_FILE = RAW_DIR / "mei_raw.txt"
PARQUET_FILE = OUT_DIR / "mei.parquet"
CSV_FILE = OUT_DIR / "mei.csv"

DATE_START = "1961-01-01"
DATE_END = "2025-12-31"

MISSING_VALUE = -999.00

# Mapeamento das 12 estações bimestrais para o "mês de referência"
# (segundo mês da janela, convenção NOAA/CPC)
BIMESTRAL_LABELS = ["DJ", "JF", "FM", "MA", "AM", "MJ",
                    "JJ", "JA", "AS", "SO", "ON", "ND"]
# DJ → 1 (Jan), JF → 2 (Fev), ..., ND → 12 (Dez)
LABEL_TO_MONTH = {label: i + 1 for i, label in enumerate(BIMESTRAL_LABELS)}

# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------


def download(url: str, dest: Path) -> Path:
    """Baixa o arquivo bruto e mantém cópia local."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando MEI.v2 de {url} ...")
    try:
        with urlopen(url, timeout=30) as resp:
            content = resp.read()
    except URLError as e:
        raise RuntimeError(f"Falha no download de {url}: {e}")

    dest.write_bytes(content)
    print(f"Arquivo bruto salvo em {dest} ({len(content)/1024:.1f} KB)")
    return dest


def parse_mei(raw_file: Path) -> pd.DataFrame:
    """Parseia o arquivo ASCII do MEI.v2 para DataFrame long format mensal."""
    print(f"Parseando {raw_file.name} ...")

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
                # rodapé textual
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

    # DataFrame com colunas DJ, JF, FM, ..., ND
    df = pd.DataFrame(rows, columns=["year"] + BIMESTRAL_LABELS)

    # Long format: (year, season_label, mei)
    df = df.melt(id_vars="year", var_name="season", value_name="mei")

    # Atribui cada estação bimestral ao seu mês de referência
    df["month"] = df["season"].map(LABEL_TO_MONTH)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))

    # Remove missing values
    df = df[df["mei"] != MISSING_VALUE].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "year", "month", "season", "mei"]]

    return df


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Recorta a série para o intervalo desejado."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


def save_outputs(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    """Persiste em CSV (sempre) e Parquet (se pyarrow disponível)."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"  CSV     : {csv_path}")

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"  Parquet : {parquet_path}")
    except ImportError:
        print(f"  Parquet : PULADO (instale com: pip install pyarrow)")
    except Exception as e:
        print(f"  Parquet : ERRO ao salvar ({type(e).__name__}: {e})")


def summary(df: pd.DataFrame) -> None:
    """Imprime resumo de qualidade dos dados."""
    print("\n--- Resumo da série MEI.v2 ---")
    print(f"  Período          : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total de meses   : {len(df)}")
    print(f"  Anos cobertos    : {df['year'].nunique()}")
    print(f"  MEI mínimo       : {df['mei'].min():+.2f}  ({df.loc[df['mei'].idxmin(), 'date'].date()})")
    print(f"  MEI máximo       : {df['mei'].max():+.2f}  ({df.loc[df['mei'].idxmax(), 'date'].date()})")
    print(f"  MEI médio        : {df['mei'].mean():+.3f}")
    print(f"  Desvio-padrão    : {df['mei'].std():.3f}")

    # Classificação NOAA (limiar |MEI| ≥ 0.5)
    n_el_nino = (df["mei"] >= 0.5).sum()
    n_la_nina = (df["mei"] <= -0.5).sum()
    n_neutro = len(df) - n_el_nino - n_la_nina
    print(f"  Meses El Niño    : {n_el_nino:4d}  (MEI ≥ +0.5)")
    print(f"  Meses La Niña    : {n_la_nina:4d}  (MEI ≤ -0.5)")
    print(f"  Meses neutros    : {n_neutro:4d}")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print(" Download e parsing do MEI.v2 (Multivariate ENSO Index v2)")
    print("=" * 70)

    raw_path = download(URL, RAW_FILE)
    df_full = parse_mei(raw_path)
    df = filter_period(df_full, DATE_START, DATE_END)

    print(f"\nApós recorte para {DATE_START} → {DATE_END}: {len(df)} meses")
    print(f"  (Lembrete: MEI.v2 só existe a partir de 1979 — anos 1961-1978 ficam ausentes)")

    save_outputs(df, PARQUET_FILE, CSV_FILE)
    summary(df)

    print("\nConcluído.")


if __name__ == "__main__":
    main()