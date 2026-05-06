"""
plot_mei_timeseries.py
======================

Gera figura de qualidade publicável (estilo Nature/Q1) da série MEI.v2
de 1979 a 2025, identificando episódios El Niño, La Niña e neutros.

Painéis:
    A — Série temporal mensal preenchida por fase (vermelho/azul/cinza)
        com anotações dos episódios mais marcantes.
    B — Histograma da distribuição de valores MEI no período.
    C — Contagem de meses por fase e intensidade (fraca/moderada/forte/muito forte).

Saídas:
    figures/mei_timeseries_1979_2025.png  (300 DPI, raster para apresentação)
    figures/mei_timeseries_1979_2025.pdf  (vetor, para publicação)

Dependências: matplotlib, pandas, numpy. Opcional: pyarrow para Parquet.

Uso:
    python plot_mei_timeseries.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

PARQUET_PATH = Path("data/oceanic_processed/mei.parquet")
CSV_PATH = Path("data/oceanic_processed/mei.csv")

OUT_DIR = Path("figures")
PNG_PATH = OUT_DIR / "mei_timeseries_1979_2025.png"
PDF_PATH = OUT_DIR / "mei_timeseries_1979_2025.pdf"

# Limiares NOAA/CPC para classificação ENSO
THRESHOLDS = {
    "weak":        0.5,   # |MEI| ≥ 0.5 → fraca
    "moderate":    1.0,   # |MEI| ≥ 1.0 → moderada
    "strong":      1.5,   # |MEI| ≥ 1.5 → forte
    "very_strong": 2.0,   # |MEI| ≥ 2.0 → muito forte
}

# Paleta (red-blue divergente, publicação-friendly)
COLOR_EL_NINO = "#c1272d"   # vermelho profundo
COLOR_LA_NINA = "#0072b2"   # azul profundo
COLOR_NEUTRAL = "#bbbbbb"   # cinza médio
COLOR_GRID = "#e5e5e5"

# Eventos marcantes (data central aproximada e rótulo)
MAJOR_EVENTS = [
    ("1982-12-01", "El Niño 1982-83",     "el_nino"),
    ("1988-09-01", "La Niña 1988-89",     "la_nina"),
    ("1997-11-01", "El Niño 1997-98",     "el_nino"),
    ("1999-12-01", "La Niña 1998-2001",   "la_nina"),
    ("2010-09-01", "La Niña 2010-11",     "la_nina"),
    ("2015-11-01", "El Niño 2015-16",     "el_nino"),
    ("2021-10-01", "La Niña 2020-23\n(triple-dip)", "la_nina"),
    ("2023-11-01", "El Niño 2023-24",     "el_nino"),
]

# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------


def load_mei() -> pd.DataFrame:
    """Carrega Parquet se disponível, senão CSV."""
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        print(f"Carregado de Parquet: {PARQUET_PATH} ({len(df)} meses)")
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        print(f"Carregado de CSV: {CSV_PATH} ({len(df)} meses)")
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo MEI encontrado.\n"
            f"  Procurei em: {PARQUET_PATH} e {CSV_PATH}\n"
            f"  Rode primeiro: python download_mei.py"
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Estatísticas
# ---------------------------------------------------------------------------


def compute_stats(df: pd.DataFrame) -> dict:
    """Calcula estatísticas operacionais da série."""
    mei = df["mei"].values
    n = len(mei)

    n_el_nino = int((mei >= THRESHOLDS["weak"]).sum())
    n_la_nina = int((mei <= -THRESHOLDS["weak"]).sum())
    n_neutral = n - n_el_nino - n_la_nina

    # Intensidade dos meses El Niño
    en_weak = int(((mei >= 0.5) & (mei < 1.0)).sum())
    en_moderate = int(((mei >= 1.0) & (mei < 1.5)).sum())
    en_strong = int(((mei >= 1.5) & (mei < 2.0)).sum())
    en_very_strong = int((mei >= 2.0).sum())

    # Intensidade dos meses La Niña
    ln_weak = int(((mei <= -0.5) & (mei > -1.0)).sum())
    ln_moderate = int(((mei <= -1.0) & (mei > -1.5)).sum())
    ln_strong = int(((mei <= -1.5) & (mei > -2.0)).sum())
    ln_very_strong = int((mei <= -2.0).sum())

    return {
        "n_total": n,
        "n_years": df["year"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "mei_min": float(mei.min()),
        "mei_max": float(mei.max()),
        "mei_mean": float(mei.mean()),
        "mei_std": float(mei.std()),
        "date_min_event": df.loc[df["mei"].idxmin(), "date"],
        "date_max_event": df.loc[df["mei"].idxmax(), "date"],
        "n_el_nino": n_el_nino,
        "n_la_nina": n_la_nina,
        "n_neutral": n_neutral,
        "pct_el_nino": 100 * n_el_nino / n,
        "pct_la_nina": 100 * n_la_nina / n,
        "pct_neutral": 100 * n_neutral / n,
        "en_intensity": (en_weak, en_moderate, en_strong, en_very_strong),
        "ln_intensity": (ln_weak, ln_moderate, ln_strong, ln_very_strong),
    }


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------


def setup_style() -> None:
    """Configura matplotlib para visual publicável."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 100,
    })


def plot_timeseries(ax, df: pd.DataFrame) -> None:
    """Painel A: série temporal preenchida por fase ENSO."""
    dates = df["date"].values
    mei = df["mei"].values

    # Fills coloridos por fase
    ax.fill_between(dates, 0, mei, where=(mei >= THRESHOLDS["weak"]),
                    interpolate=True, color=COLOR_EL_NINO, alpha=0.85,
                    linewidth=0, label=f"El Niño (MEI ≥ +{THRESHOLDS['weak']})")
    ax.fill_between(dates, 0, mei, where=(mei <= -THRESHOLDS["weak"]),
                    interpolate=True, color=COLOR_LA_NINA, alpha=0.85,
                    linewidth=0, label=f"La Niña (MEI ≤ -{THRESHOLDS['weak']})")
    # Faixa neutra (entre -0.5 e +0.5)
    neutral_mask = np.abs(mei) < THRESHOLDS["weak"]
    ax.fill_between(dates, 0, mei, where=neutral_mask,
                    interpolate=True, color=COLOR_NEUTRAL, alpha=0.65,
                    linewidth=0, label="Neutro (|MEI| < 0.5)")

    # Linha mestra
    ax.plot(dates, mei, color="black", linewidth=0.6, alpha=0.9)

    # Linhas de referência
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.7)
    for level, ls, lw in [(0.5, ":", 0.6), (1.0, "--", 0.5),
                          (1.5, "-.", 0.5), (2.0, ":", 0.4)]:
        ax.axhline( level, color="#777", linewidth=lw, linestyle=ls, alpha=0.55)
        ax.axhline(-level, color="#777", linewidth=lw, linestyle=ls, alpha=0.55)

    # Anotações dos eventos
    y_offsets_pos = {"el_nino": 1, "la_nina": -1}
    for date_str, label, kind in MAJOR_EVENTS:
        dt = pd.Timestamp(date_str)
        if dt < df["date"].min() or dt > df["date"].max():
            continue
        # Pega o valor MEI mais próximo dessa data
        idx = (df["date"] - dt).abs().idxmin()
        y_event = df.loc[idx, "mei"]

        if kind == "el_nino":
            xytext_offset = (0, 28)
            color = COLOR_EL_NINO
        else:
            xytext_offset = (0, -32)
            color = COLOR_LA_NINA

        ax.annotate(
            label,
            xy=(dt, y_event),
            xytext=xytext_offset, textcoords="offset points",
            ha="center", va="center",
            fontsize=8, color=color, fontweight="semibold",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, linewidth=0.6),
        )

    # Eixos e labels
    ax.set_ylabel("MEI.v2 (índice padronizado)")
    ax.set_xlabel("Ano")
    ax.set_xlim(df["date"].min(), df["date"].max())
    ymin = min(-3, mei.min() - 0.3)
    ymax = max( 3, mei.max() + 0.3)
    ax.set_ylim(ymin, ymax)

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_title("A. Série temporal MEI.v2 (1979-2025) — episódios ENSO mensais",
                 loc="left")

    # Legenda customizada
    legend_handles = [
        Patch(facecolor=COLOR_EL_NINO, alpha=0.85, label=f"El Niño  (MEI ≥ +0.5)"),
        Patch(facecolor=COLOR_LA_NINA, alpha=0.85, label=f"La Niña  (MEI ≤ -0.5)"),
        Patch(facecolor=COLOR_NEUTRAL, alpha=0.65, label="Neutro    (|MEI| < 0.5)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.005, 0.99), ncol=3, frameon=False)


def plot_histogram(ax, df: pd.DataFrame) -> None:
    """Painel B: histograma da distribuição de valores."""
    mei = df["mei"].values

    bins = np.arange(np.floor(mei.min() * 2) / 2,
                     np.ceil(mei.max() * 2) / 2 + 0.25, 0.25)
    counts, edges = np.histogram(mei, bins=bins)

    # Cor por bin baseada no centro do bin
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [COLOR_EL_NINO if c >= 0.5 else
              (COLOR_LA_NINA if c <= -0.5 else COLOR_NEUTRAL)
              for c in centers]

    ax.bar(centers, counts, width=0.23, color=colors, edgecolor="white",
           linewidth=0.4, alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.6, alpha=0.7)
    ax.axvline( 0.5, color="#777", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axvline(-0.5, color="#777", linewidth=0.6, linestyle=":", alpha=0.7)

    ax.set_xlabel("MEI.v2")
    ax.set_ylabel("Frequência (meses)")
    ax.set_title("B. Distribuição", loc="left")

    # Estatística sobreposta
    mu = mei.mean()
    sigma = mei.std()
    txt = f"μ = {mu:+.3f}\nσ = {sigma:.3f}\nn = {len(mei)}"
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#bbb", alpha=0.9))


def plot_intensity_bars(ax, stats: dict) -> None:
    """Painel C: contagem de meses por fase e intensidade."""
    en = stats["en_intensity"]    # (weak, moderate, strong, very_strong)
    ln = stats["ln_intensity"]
    neutral = stats["n_neutral"]

    categories = ["Muito\nforte\n(≥2.0)", "Forte\n(≥1.5)", "Moderada\n(≥1.0)",
                  "Fraca\n(≥0.5)", "Neutro\n(<0.5)",
                  "Fraca\n(≥0.5)", "Moderada\n(≥1.0)", "Forte\n(≥1.5)",
                  "Muito\nforte\n(≥2.0)"]
    counts = [ln[3], ln[2], ln[1], ln[0], neutral, en[0], en[1], en[2], en[3]]
    colors = [COLOR_LA_NINA] * 4 + [COLOR_NEUTRAL] + [COLOR_EL_NINO] * 4
    alphas = [0.55, 0.70, 0.85, 1.00, 0.65, 1.00, 0.85, 0.70, 0.55]

    x = np.arange(len(counts))
    bars = ax.bar(x, counts, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # Rótulos com contagem em cada barra
    for xi, c in zip(x, counts):
        if c > 0:
            ax.text(xi, c + max(counts) * 0.02, str(c),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel("Número de meses")
    ax.set_ylim(0, max(counts) * 1.18)

    # Faixas de fundo para indicar grupo
    ax.axvspan(-0.5, 3.5, color=COLOR_LA_NINA, alpha=0.07, zorder=0)
    ax.axvspan(3.5, 4.5, color=COLOR_NEUTRAL, alpha=0.10, zorder=0)
    ax.axvspan(4.5, 8.5, color=COLOR_EL_NINO, alpha=0.07, zorder=0)

    # Rótulos de grupo no topo
    ax.text(1.5, max(counts) * 1.13, "La Niña", ha="center",
            color=COLOR_LA_NINA, fontweight="bold", fontsize=9)
    ax.text(4.0, max(counts) * 1.13, "Neutro", ha="center",
            color="#666", fontweight="bold", fontsize=9)
    ax.text(6.5, max(counts) * 1.13, "El Niño", ha="center",
            color=COLOR_EL_NINO, fontweight="bold", fontsize=9)

    ax.set_title("C. Distribuição de meses por fase e intensidade", loc="left")


def add_summary_text(fig, stats: dict) -> None:
    """Caixa de texto com resumo estatístico no canto inferior."""
    txt = (
        f"Período analisado: {stats['date_min'].strftime('%b/%Y')} – "
        f"{stats['date_max'].strftime('%b/%Y')}  ({stats['n_total']} meses, "
        f"{stats['n_years']} anos)\n"
        f"MEI mínimo: {stats['mei_min']:+.2f} "
        f"({stats['date_min_event'].strftime('%b/%Y')}, La Niña 2010-11)    "
        f"MEI máximo: {stats['mei_max']:+.2f} "
        f"({stats['date_max_event'].strftime('%b/%Y')})\n"
        f"El Niño: {stats['n_el_nino']} meses ({stats['pct_el_nino']:.1f}%)    "
        f"La Niña: {stats['n_la_nina']} meses ({stats['pct_la_nina']:.1f}%)    "
        f"Neutro: {stats['n_neutral']} meses ({stats['pct_neutral']:.1f}%)"
    )
    fig.text(0.5, 0.02, txt, ha="center", va="bottom", fontsize=8.5,
             color="#333",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7",
                       edgecolor="#ccc"))


def make_figure(df: pd.DataFrame, stats: dict) -> plt.Figure:
    """Monta a figura multi-painel completa."""
    setup_style()

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.3, 1.0],
        width_ratios=[1.0, 1.4],
        hspace=0.45, wspace=0.28,
        left=0.07, right=0.97, top=0.93, bottom=0.13,
    )

    ax_main = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_bars = fig.add_subplot(gs[1, 1])

    plot_timeseries(ax_main, df)
    plot_histogram(ax_hist, df)
    plot_intensity_bars(ax_bars, stats)

    # Título superior
    fig.suptitle(
        "Multivariate ENSO Index (MEI.v2) — episódios El Niño, La Niña e neutros",
        fontsize=13, fontweight="bold", y=0.985,
    )

    add_summary_text(fig, stats)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print(" Geração de figura: série temporal MEI.v2 (1979-2025)")
    print("=" * 70)

    df = load_mei()
    stats = compute_stats(df)

    print(f"\nResumo:")
    print(f"  Período           : {stats['date_min'].date()} → {stats['date_max'].date()}")
    print(f"  Total de meses    : {stats['n_total']}")
    print(f"  Meses El Niño     : {stats['n_el_nino']:4d}  ({stats['pct_el_nino']:.1f}%)")
    print(f"  Meses La Niña     : {stats['n_la_nina']:4d}  ({stats['pct_la_nina']:.1f}%)")
    print(f"  Meses neutros     : {stats['n_neutral']:4d}  ({stats['pct_neutral']:.1f}%)")
    print(f"  MEI mínimo        : {stats['mei_min']:+.2f}  ({stats['date_min_event'].date()})")
    print(f"  MEI máximo        : {stats['mei_max']:+.2f}  ({stats['date_max_event'].date()})")

    print("\nGerando figura ...")
    fig = make_figure(df, stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  PNG salvo : {PNG_PATH}")
    try:
        fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
        print(f"  PDF salvo : {PDF_PATH}")
    except Exception as e:
        print(f"  PDF       : ERRO ({type(e).__name__}: {e})")

    plt.close(fig)
    print("\nConcluído.")


if __name__ == "__main__":
    main()