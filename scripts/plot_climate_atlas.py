"""
plot_climate_atlas.py
=====================

Gera figura de caracterização climática multi-painel das 10 cidades do
MATOPIBA, período 1961-2025, no padrão de revista Q1 (Agricultural and
Forest Meteorology, Journal of Hydrology, etc.).

Painéis:
    A — Precipitação anual: série temporal por cidade + ensemble mediano +
        envelope interquartílico + tendência Mann-Kendall/Sen.
    B — Climatologia mensal de precipitação: ciclo médio + variabilidade
        entre cidades.
    C — Evapotranspiração de referência (ET₀) anual: análoga ao painel A.
    D — Climatologia mensal de ET₀: análoga ao painel B.

Saídas:
    figures/climate_atlas_matopiba_1961_2025.png
    figures/climate_atlas_matopiba_1961_2025.pdf

Estatísticas reportadas:
    - Tendência de Sen (Sen's slope) — robusto a outliers, não paramétrico.
    - Mann-Kendall p-value — teste não paramétrico padrão em hidroclimatologia.
    - Período de referência climatológico: WMO 1991-2020 (ou customizável).

Dependências: matplotlib, pandas, numpy, scipy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Sobe na hierarquia de diretórios até encontrar a raiz do projeto.

    Procura por marcadores típicos de raiz: a pasta 'data/extracted_csv/'
    ou um diretório '.git'. Fallback: diretório do próprio script.
    """
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / "data" / "extracted_csv").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return here


PROJECT_ROOT = _find_project_root()

DATA_DIR = PROJECT_ROOT / "data" / "extracted_csv" / "merged_by_city"
OUT_DIR = PROJECT_ROOT / "figures"
PNG_PATH = OUT_DIR / "climate_atlas_matopiba_1961_2025.png"
PDF_PATH = OUT_DIR / "climate_atlas_matopiba_1961_2025.pdf"

YEAR_START = 1961
YEAR_END = 2025

# Período de referência climatológico (norma WMO 30 anos)
CLIM_REF_START = 1991
CLIM_REF_END = 2020

# Cidades MATOPIBA na ordem em que vão aparecer na legenda
CITIES = [
    "Baixa_Grande_do_Ribeiro",
    "Balsas",
    "Barreiras",
    "Bom_Jesus",
    "Campos_Lindos",
    "Correntina",
    "Formosa_do_Rio_Preto",
    "Luis_Eduardo_Magalhaes",
    "Tasso_Fragoso",
    "Urucui",
]

# Paleta categórica perceptualmente uniforme (10 cidades, colorblind-safe)
CITY_COLORS = plt.cm.tab10(np.linspace(0, 1, len(CITIES)))

# Cores semânticas
COLOR_ENSEMBLE = "#222222"      # mediana ensemble
COLOR_TREND = "#c1272d"         # linha de tendência
COLOR_ENVELOPE = "#888888"      # banda IQR
COLOR_CLIM_REF = "#0072b2"      # média climatológica de referência

# Aggregação por variável (sum vs mean)
AGG_RULES = {
    "pr":   "sum",   # precipitação total (mm/ano ou mm/mês)
    "ETo":  "sum",   # ET₀ acumulada (mm/ano ou mm/mês)
    "Tmax": "mean",  # temperatura máxima média (°C)
    "Tmin": "mean",  # temperatura mínima média (°C)
    "RH":   "mean",  # umidade relativa média (%)
    "Rs":   "mean",  # radiação solar média (MJ/m²/dia)
    "u2":   "mean",  # vento médio (m/s)
}

VAR_LABELS = {
    "pr":   "Precipitação (mm/ano)",
    "ETo":  "ET₀ de referência (mm/ano)",
    "Tmax": "Temperatura máxima (°C)",
    "Tmin": "Temperatura mínima (°C)",
}

VAR_LABELS_MONTHLY = {
    "pr":   "Precipitação (mm/mês)",
    "ETo":  "ET₀ (mm/mês)",
}

# ---------------------------------------------------------------------------
# Carregamento e agregação
# ---------------------------------------------------------------------------


def load_city(city: str) -> pd.DataFrame:
    """Carrega o CSV diário de uma cidade."""
    path = DATA_DIR / f"{city}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    df = df.loc[f"{YEAR_START}-01-01":f"{YEAR_END}-12-31"]
    return df


def annual_aggregate(df: pd.DataFrame, var: str) -> pd.Series:
    """Agrega série diária para anual (sum ou mean conforme variável)."""
    rule = AGG_RULES[var]
    by_year = df[var].groupby(df.index.year)
    if rule == "sum":
        return by_year.sum()
    return by_year.mean()


def monthly_climatology(df: pd.DataFrame, var: str) -> pd.Series:
    """Ciclo climatológico mensal médio (12 valores)."""
    rule = AGG_RULES[var]
    # Primeiro agrega por (ano, mês), depois tira média mensal interanual
    by_ym = df[var].groupby([df.index.year, df.index.month])
    monthly = by_ym.sum() if rule == "sum" else by_ym.mean()
    monthly_clim = monthly.groupby(level=1).mean()
    monthly_clim.index.name = "month"
    return monthly_clim


def load_all_cities() -> dict[str, pd.DataFrame]:
    """Carrega CSVs de todas as cidades em um dict."""
    print(f"Carregando {len(CITIES)} cidades de {DATA_DIR} ...")
    data = {}
    for city in CITIES:
        try:
            data[city] = load_city(city)
            n_days = len(data[city])
            n_vars = len(data[city].columns)
            print(f"  {city:32s}  {n_days:6d} dias, {n_vars} vars")
        except FileNotFoundError as e:
            print(f"  {city:32s}  ERRO: {e}")
    return data


# ---------------------------------------------------------------------------
# Estatística de tendência (Mann-Kendall + Sen's slope)
# ---------------------------------------------------------------------------


def sens_slope(years: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Estimador robusto de Sen para tendência linear.

    Returns
    -------
    slope, intercept, slope_std
    """
    n = len(years)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((values[j] - values[i]) / (years[j] - years[i]))
    slopes = np.array(slopes)
    slope = np.median(slopes)
    intercept = np.median(values - slope * years)
    slope_std = stats.median_abs_deviation(slopes) * 1.4826  # MAD-based std
    return slope, intercept, slope_std


def mann_kendall_test(values: np.ndarray) -> tuple[float, float]:
    """Teste de Mann-Kendall (não paramétrico) para tendência monotônica.

    Returns
    -------
    tau, p_value
    """
    tau, p_value = stats.kendalltau(np.arange(len(values)), values)
    return tau, p_value


def trend_summary(years: np.ndarray, values: np.ndarray) -> dict:
    """Resumo completo de tendência (Sen + Mann-Kendall)."""
    slope, intercept, _ = sens_slope(years, values)
    tau, p = mann_kendall_test(values)
    return {
        "slope_per_year": slope,
        "slope_per_decade": slope * 10,
        "intercept": intercept,
        "tau": tau,
        "p_value": p,
        "significant_05": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------


def setup_style() -> None:
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
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 100,
    })


# ---------------------------------------------------------------------------
# Painéis
# ---------------------------------------------------------------------------


def plot_annual_panel(ax, data: dict, var: str, panel_label: str) -> None:
    """Painel de série temporal anual com ensemble e tendência."""
    series_dict = {}
    for city, df in data.items():
        series_dict[city] = annual_aggregate(df, var)

    annual_df = pd.DataFrame(series_dict)
    years = annual_df.index.values

    # Linhas individuais por cidade (finas, semi-transparentes)
    for i, city in enumerate(CITIES):
        if city not in annual_df.columns:
            continue
        ax.plot(years, annual_df[city], color=CITY_COLORS[i],
                linewidth=0.7, alpha=0.55,
                label=city.replace("_", " "))

    # Mediana entre cidades (ensemble)
    ensemble_med = annual_df.median(axis=1)
    ensemble_q25 = annual_df.quantile(0.25, axis=1)
    ensemble_q75 = annual_df.quantile(0.75, axis=1)

    ax.fill_between(years, ensemble_q25, ensemble_q75,
                    color=COLOR_ENVELOPE, alpha=0.18,
                    label="IQR entre cidades")
    ax.plot(years, ensemble_med, color=COLOR_ENSEMBLE, linewidth=1.6,
            label="Mediana ensemble")

    # Tendência de Sen no ensemble mediano
    valid = ensemble_med.notna()
    yrs_v = years[valid]
    val_v = ensemble_med.values[valid]
    trend = trend_summary(yrs_v, val_v)
    trend_line = trend["intercept"] + trend["slope_per_year"] * yrs_v
    sig = "***" if trend["p_value"] < 0.001 else (
          "**" if trend["p_value"] < 0.01 else (
          "*" if trend["p_value"] < 0.05 else "n.s."))
    ax.plot(yrs_v, trend_line, color=COLOR_TREND, linewidth=1.4,
            linestyle="--", alpha=0.9,
            label=(f"Tendência Sen: {trend['slope_per_decade']:+.1f}/década ({sig})"))

    # Média climatológica de referência (WMO 1991-2020)
    ref_mask = (annual_df.index >= CLIM_REF_START) & (annual_df.index <= CLIM_REF_END)
    clim_ref = annual_df.loc[ref_mask].median(axis=1).mean()
    ax.axhline(clim_ref, color=COLOR_CLIM_REF, linewidth=0.8,
               linestyle=":", alpha=0.8,
               label=f"Média {CLIM_REF_START}-{CLIM_REF_END}: {clim_ref:.0f}")

    ax.set_xlabel("Ano")
    ax.set_ylabel(VAR_LABELS.get(var, var))
    ax.set_xlim(YEAR_START, YEAR_END)

    ax.set_title(f"{panel_label}. {VAR_LABELS.get(var, var)} — séries anuais (1961-2025)",
                 loc="left")

    # Texto com estatísticas
    stat_txt = (
        f"Tendência (Sen, mediana): {trend['slope_per_decade']:+.2f} "
        f"unidades/década\n"
        f"Mann-Kendall: τ = {trend['tau']:+.3f}, p = {trend['p_value']:.4f}\n"
        f"Período: {YEAR_START}-{YEAR_END} ({YEAR_END - YEAR_START + 1} anos)"
    )
    ax.text(0.02, 0.97, stat_txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#bbb", alpha=0.9))


def plot_monthly_panel(ax, data: dict, var: str, panel_label: str) -> None:
    """Painel de ciclo climatológico mensal."""
    monthly_dict = {}
    for city, df in data.items():
        monthly_dict[city] = monthly_climatology(df, var)

    monthly_df = pd.DataFrame(monthly_dict)
    months = np.arange(1, 13)

    # Linhas individuais por cidade
    for i, city in enumerate(CITIES):
        if city not in monthly_df.columns:
            continue
        ax.plot(months, monthly_df[city], color=CITY_COLORS[i],
                linewidth=0.7, alpha=0.55, marker="o", markersize=2.5)

    # Ensemble mediano + IQR
    ens_med = monthly_df.median(axis=1)
    ens_q25 = monthly_df.quantile(0.25, axis=1)
    ens_q75 = monthly_df.quantile(0.75, axis=1)

    ax.fill_between(months, ens_q25, ens_q75, color=COLOR_ENVELOPE, alpha=0.20)
    ax.plot(months, ens_med, color=COLOR_ENSEMBLE, linewidth=1.8,
            marker="s", markersize=4)

    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                        "Jul", "Ago", "Set", "Out", "Nov", "Dez"], fontsize=8)
    ax.set_xlim(0.5, 12.5)
    ax.set_xlabel("Mês")
    ax.set_ylabel(VAR_LABELS_MONTHLY.get(var, var))

    ax.set_title(f"{panel_label}. Ciclo climatológico mensal "
                 f"({VAR_LABELS_MONTHLY.get(var, var).split(' (')[0]})",
                 loc="left")

    # Identificar estação chuvosa vs seca
    if var == "pr":
        wet_months = ens_med[ens_med > ens_med.mean()].index.tolist()
        if wet_months:
            wet_start = min(wet_months)
            wet_end = max(wet_months)
            ax.axvspan(wet_start - 0.5, wet_end + 0.5,
                       color="#0072b2", alpha=0.06, zorder=0)
            ax.text(0.5 * (wet_start + wet_end), ax.get_ylim()[1] * 0.92,
                    "Estação chuvosa", ha="center", va="center",
                    fontsize=8, color="#0072b2", style="italic")


# ---------------------------------------------------------------------------
# Figura completa
# ---------------------------------------------------------------------------


def make_figure(data: dict) -> plt.Figure:
    setup_style()

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.6, 1.0],
        hspace=0.40, wspace=0.22,
        left=0.06, right=0.97, top=0.93, bottom=0.17,
    )

    ax_pr_annual = fig.add_subplot(gs[0, 0])
    ax_pr_monthly = fig.add_subplot(gs[0, 1])
    ax_eto_annual = fig.add_subplot(gs[1, 0])
    ax_eto_monthly = fig.add_subplot(gs[1, 1])

    plot_annual_panel(ax_pr_annual, data, "pr", "A")
    plot_monthly_panel(ax_pr_monthly, data, "pr", "B")
    plot_annual_panel(ax_eto_annual, data, "ETo", "C")
    plot_monthly_panel(ax_eto_monthly, data, "ETo", "D")

    # Legenda compartilhada na parte inferior, em formato horizontal.
    # Com 14 itens (10 cidades + 4 elementos estatísticos), usar 7 colunas
    # produz 2 linhas equilibradas; 5 colunas produzem 3 linhas mais arejadas.
    handles, labels = ax_pr_annual.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=5,
               fontsize=8.5,
               title="Cidades MATOPIBA + elementos estatísticos do ensemble",
               title_fontsize=10, frameon=False,
               columnspacing=1.6, handlelength=2.2)

    fig.suptitle(
        "Caracterização climatológica das 10 cidades do MATOPIBA "
        "(1961-2025)\nFonte: Xavier BR-DWGD",
        fontsize=13, fontweight="bold", y=0.985,
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print(" Climate atlas MATOPIBA 1961-2025")
    print("=" * 70)
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Output dir    : {OUT_DIR}")
    print()

    data = load_all_cities()

    if not data:
        print("\nNenhum dado carregado. Verifique o caminho em DATA_DIR.")
        return

    print(f"\nGerando figura ({len(data)} cidades) ...")
    fig = make_figure(data)

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