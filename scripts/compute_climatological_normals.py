"""
compute_climatological_normals.py
==================================

Calcula normais climatológicas para as 10 cidades MATOPIBA nos três períodos
de referência padrão da WMO (1961-1990, 1981-2010, 1991-2020), aplica testes
estatísticos de diferença entre períodos e gera planilha Excel multi-aba +
tabelas LaTeX prontas para o paper.

Estatísticas calculadas (por cidade × variável × período):
    - Tendência central : mean, median
    - Dispersão         : std, CV, IQR
    - Quantis           : P10, P25, P50, P75, P90
    - Forma             : skewness, kurtosis
    - Extremos          : min, max
    - Suficiência       : n (tamanho amostral efetivo)

Testes aplicados (entre os três períodos):
    - Kruskal-Wallis    : teste global de diferença entre 3 períodos
    - Mann-Whitney U    : pairwise (1961-90 vs 1981-2010 vs 1991-2020)

Saídas:
    data_processed/climatological_normals/
        ├── normals_annual.xlsx           — estatísticas anuais (3 períodos)
        ├── normals_monthly.xlsx          — climatologia mensal (3 períodos)
        ├── period_comparison_tests.xlsx  — testes estatísticos
        └── latex_tables/
            ├── table_annual_normals.tex
            └── table_period_comparison.tex

Uso:
    python scripts/compute_climatological_normals.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Suprime avisos de scipy com amostras pequenas (esperados em alguns meses)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Localiza a raiz do projeto procurando por marcadores típicos."""
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / "data" / "extracted_csv").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return here


PROJECT_ROOT = _find_project_root()

DATA_DIR = PROJECT_ROOT / "data" / "extracted_csv" / "merged_by_city"
OUT_DIR = PROJECT_ROOT / "data_processed" / "climatological_normals"
LATEX_DIR = OUT_DIR / "latex_tables"

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

# Períodos OMM (WMO Climatological Standard Normals)
WMO_PERIODS = {
    "1961-1990": (1961, 1990),
    "1981-2010": (1981, 2010),
    "1991-2020": (1991, 2020),
}

# Regras de agregação por variável (ver FAO-56 / convenção hidroclimática)
AGG_RULES = {
    "pr":   "sum",   # precipitação total
    "ETo":  "sum",   # evapotranspiração de referência total
    "Tmax": "mean",  # temperatura máxima média
    "Tmin": "mean",  # temperatura mínima média
    "RH":   "mean",  # umidade relativa média
    "Rs":   "mean",  # radiação solar média
    "u2":   "mean",  # vento médio
}

VAR_UNITS = {
    "pr":   "mm",
    "ETo":  "mm",
    "Tmax": "°C",
    "Tmin": "°C",
    "RH":   "%",
    "Rs":   "MJ/m²/d",
    "u2":   "m/s",
}

VAR_LABELS = {
    "pr":   "Precipitation (P)",
    "ETo":  "Reference ET (ET0)",
    "Tmax": "Max temperature (Tmax)",
    "Tmin": "Min temperature (Tmin)",
    "RH":   "Relative humidity (RH)",
    "Rs":   "Solar radiation (Rs)",
    "u2":   "Wind speed (u2)",
}

# ---------------------------------------------------------------------------
# Carregamento e agregação
# ---------------------------------------------------------------------------


def load_city(city: str) -> pd.DataFrame:
    """Carrega o CSV diário de uma cidade (1961-2025)."""
    path = DATA_DIR / f"{city}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()


def annual_series(df: pd.DataFrame, var: str) -> pd.Series:
    """Agregação anual da variável (sum ou mean conforme AGG_RULES).

    Retorna Series vazia (dtype=float) se a coluna não existir no DataFrame.
    Isso permite que o pipeline continue executando mesmo se uma cidade
    estiver com dataset incompleto (ex.: ETo ainda não baixado).
    """
    if var not in df.columns:
        return pd.Series(dtype=float)
    rule = AGG_RULES[var]
    by_year = df[var].groupby(df.index.year)
    return by_year.sum() if rule == "sum" else by_year.mean()


def monthly_dataframe(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Série mensal como DataFrame plano com colunas (year, month, value).

    Evita MultiIndex porque pandas é ordens de magnitude mais rápido em
    filtragem de DataFrame regular. Esta função é chamada apenas uma vez
    por (cidade, variável); todas as filtragens posteriores por período/mês
    são vetorizadas e baratas.

    Retorna DataFrame vazio se a coluna não existir.

    Nota metodológica: para variáveis com agregação por soma (precipitação,
    ET₀), o resample soma APENAS os dias disponíveis no mês. Em meses com
    dados faltantes (gaps no Xavier), o total mensal fica subestimado. Esta
    é a convenção climatológica padrão (não se preenche sinteticamente);
    o tamanho amostral efetivo é registrado em `n` nas estatísticas.
    """
    if var not in df.columns:
        return pd.DataFrame(columns=["year", "month", "value"])
    rule = AGG_RULES[var]
    monthly = df[var].resample("MS").sum() if rule == "sum" else df[var].resample("MS").mean()
    return pd.DataFrame({
        "year":  monthly.index.year,
        "month": monthly.index.month,
        "value": monthly.values,
    })


def filter_period(s: pd.Series, year_start: int, year_end: int) -> pd.Series:
    """Filtra Series indexada por ano (para séries anuais)."""
    mask = (s.index >= year_start) & (s.index <= year_end)
    return s.loc[mask]


# ---------------------------------------------------------------------------
# Cálculo de estatísticas
# ---------------------------------------------------------------------------


def descriptive_stats(values: np.ndarray) -> dict:
    """Conjunto canônico de estatísticas descritivas para climatologia."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {k: np.nan for k in [
            "n", "mean", "median", "std", "cv", "iqr",
            "p10", "p25", "p50", "p75", "p90",
            "skewness", "kurtosis", "min", "max",
        ]}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return {
        "n":        int(len(values)),
        "mean":     mean,
        "median":   float(np.median(values)),
        "std":      std,
        "cv":       float(std / mean * 100) if (mean != 0 and not np.isnan(std)) else np.nan,
        "iqr":      float(np.percentile(values, 75) - np.percentile(values, 25)),
        "p10":      float(np.percentile(values, 10)),
        "p25":      float(np.percentile(values, 25)),
        "p50":      float(np.percentile(values, 50)),
        "p75":      float(np.percentile(values, 75)),
        "p90":      float(np.percentile(values, 90)),
        "skewness": float(stats.skew(values, bias=False)) if len(values) > 2 else np.nan,
        "kurtosis": float(stats.kurtosis(values, bias=False)) if len(values) > 3 else np.nan,
        "min":      float(np.min(values)),
        "max":      float(np.max(values)),
    }


def compute_annual_normals_table(data: dict) -> pd.DataFrame:
    """Tabela: cidade × variável × período → estatísticas anuais."""
    rows = []
    for city, df in data.items():
        for var in AGG_RULES:
            ann = annual_series(df, var)
            for period_label, (y0, y1) in WMO_PERIODS.items():
                vals = filter_period(ann, y0, y1).values
                stats_dict = descriptive_stats(vals)
                rows.append({
                    "city": city,
                    "variable": var,
                    "period": period_label,
                    **stats_dict,
                })
    return pd.DataFrame(rows)


def compute_monthly_normals_table(data: dict) -> pd.DataFrame:
    """Tabela: cidade × variável × período × mês → estatísticas mensais.

    Implementação otimizada: monthly_dataframe é calculado uma única vez
    por (cidade, variável); filtragens por período e mês são vetorizadas
    via boolean indexing em DataFrame regular (não MultiIndex).
    """
    rows = []
    n_cities = len(data)
    for ic, (city, df) in enumerate(data.items(), 1):
        for var in AGG_RULES:
            mon_df = monthly_dataframe(df, var)
            for period_label, (y0, y1) in WMO_PERIODS.items():
                mon_p = mon_df[(mon_df["year"] >= y0) & (mon_df["year"] <= y1)]
                # agrupa por mês: distribuição interanual de cada mês calendárico
                for month in range(1, 13):
                    vals = mon_p.loc[mon_p["month"] == month, "value"].values
                    stats_dict = descriptive_stats(vals)
                    rows.append({
                        "city": city,
                        "variable": var,
                        "period": period_label,
                        "month": month,
                        **stats_dict,
                    })
        print(f"  [{ic:2d}/{n_cities}] {city} processada", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Testes estatísticos entre períodos
# ---------------------------------------------------------------------------


def kruskal_three_periods(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> dict:
    """Kruskal-Wallis: teste global não paramétrico de diferença entre 3 grupos.

    H0: as três distribuições são idênticas.
    H1: pelo menos uma difere das outras (sem especificar qual).
    """
    p1 = p1[~np.isnan(p1)]
    p2 = p2[~np.isnan(p2)]
    p3 = p3[~np.isnan(p3)]
    if min(len(p1), len(p2), len(p3)) < 5:
        return {"H": np.nan, "p_value": np.nan, "significant_05": False}
    H, p = stats.kruskal(p1, p2, p3)
    return {
        "H": float(H),
        "p_value": float(p),
        "significant_05": bool(p < 0.05),
    }


def mann_whitney_pairwise(a: np.ndarray, b: np.ndarray) -> dict:
    """Mann-Whitney U para comparação de dois períodos (rank-based).

    H0: P(X > Y) = 0.5, ou seja, as duas distribuições são iguais.
    O teste é sobre distribuições/medianas, não sobre médias. Por isso
    reportamos `delta_median` como tamanho de efeito principal, e
    `delta_mean` apenas como referência adicional.
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if min(len(a), len(b)) < 5:
        return {
            "U": np.nan, "p_value": np.nan, "significant_05": False,
            "delta_median": np.nan, "delta_mean": np.nan,
        }
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "U": float(U),
        "p_value": float(p),
        "significant_05": bool(p < 0.05),
        "delta_median": float(np.median(b) - np.median(a)),
        "delta_mean":   float(np.mean(b) - np.mean(a)),
    }


def bonferroni_correct(p_values: list, n_tests: int) -> list:
    """Correção de Bonferroni para múltiplos testes.

    p_corrected = min(1.0, p × n_tests). Conservadora porém defensável.
    Para testes mais poderosos, ver Benjamini-Hochberg (False Discovery Rate).
    """
    return [min(1.0, p * n_tests) if not pd.isna(p) else np.nan for p in p_values]


def compute_period_comparison_tests(data: dict, apply_bonferroni: bool = True) -> pd.DataFrame:
    """Para cada cidade × variável: testes Kruskal-Wallis e pairwise.

    Parameters
    ----------
    data : dict
        {city: DataFrame} — séries diárias por cidade.
    apply_bonferroni : bool
        Se True, aplica correção de Bonferroni nos p-valores Kruskal-Wallis
        considerando o total de testes (cidades × variáveis). Os p-valores
        brutos também são preservados para transparência.

    Notas
    -----
    - O teste de Kruskal-Wallis avalia diferença em distribuições/medianas
      entre os 3 períodos OMM, não em médias.
    - Os p-valores pairwise (Mann-Whitney U) NÃO são ajustados para
      multiplicidade dentro de cada cidade × variável (3 comparações por
      teste). Reviewer estatístico cuidadoso pode pedir correção; o
      protocolo recomendado é Holm-Bonferroni ou Benjamini-Hochberg
      sobre todos os pairwise reportados.
    - Tamanho de efeito reportado: `delta_median` (consistente com a
      hipótese sob teste) e `delta_mean` (referência adicional).
    """
    rows = []
    for city, df in data.items():
        for var in AGG_RULES:
            ann = annual_series(df, var)
            if ann.empty:
                continue
            p1 = filter_period(ann, *WMO_PERIODS["1961-1990"]).values
            p2 = filter_period(ann, *WMO_PERIODS["1981-2010"]).values
            p3 = filter_period(ann, *WMO_PERIODS["1991-2020"]).values

            kw = kruskal_three_periods(p1, p2, p3)
            mw_12 = mann_whitney_pairwise(p1, p2)
            mw_13 = mann_whitney_pairwise(p1, p3)
            mw_23 = mann_whitney_pairwise(p2, p3)

            rows.append({
                "city":              city,
                "variable":          var,
                "kruskal_H":         kw["H"],
                "kruskal_p_raw":     kw["p_value"],
                "kruskal_sig_raw":   kw["significant_05"],
                "mw_1961-1990_vs_1981-2010_p":  mw_12["p_value"],
                "mw_1961-1990_vs_1991-2020_p":  mw_13["p_value"],
                "mw_1981-2010_vs_1991-2020_p":  mw_23["p_value"],
                "delta_median_p3_minus_p1":     mw_13["delta_median"],
                "delta_mean_p3_minus_p1":       mw_13["delta_mean"],
            })

    df_tests = pd.DataFrame(rows)
    if df_tests.empty:
        return df_tests

    if apply_bonferroni and len(df_tests) > 0:
        n = len(df_tests)
        df_tests["kruskal_p_bonferroni"] = bonferroni_correct(
            df_tests["kruskal_p_raw"].tolist(), n)
        df_tests["kruskal_sig_bonferroni"] = df_tests["kruskal_p_bonferroni"] < 0.05

    return df_tests


# ---------------------------------------------------------------------------
# Saída — Excel
# ---------------------------------------------------------------------------


def save_parquet(annual: pd.DataFrame, monthly: pd.DataFrame, tests: pd.DataFrame) -> None:
    """Formato canônico do pipeline. O modelo bayesiano lê daqui (mais rápido)."""
    parquet_dir = OUT_DIR / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    try:
        annual.to_parquet(parquet_dir / "annual_normals.parquet", index=False)
        monthly.to_parquet(parquet_dir / "monthly_normals.parquet", index=False)
        tests.to_parquet(parquet_dir / "period_tests.parquet", index=False)
        print(f"  Parquet  : {parquet_dir} (3 arquivos)")
    except ImportError:
        print(f"  Parquet  : PULADO (instale com: pip install pyarrow)")


def save_csv(annual: pd.DataFrame, monthly: pd.DataFrame, tests: pd.DataFrame) -> None:
    """Versão amigável ao Git diff e à inspeção humana rápida."""
    csv_dir = OUT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    annual.to_csv(csv_dir / "annual_normals.csv", index=False, float_format="%.3f")
    monthly.to_csv(csv_dir / "monthly_normals.csv", index=False, float_format="%.3f")
    tests.to_csv(csv_dir / "period_tests.csv", index=False, float_format="%.4f")
    print(f"  CSV      : {csv_dir} (3 arquivos)")


def save_json_priors(annual: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """JSON estruturado e aninhado — formato natural para alimentar priors PyMC.

    Estrutura:
        {
          "metadata": {...},
          "priors": {
            <city>: {
              <period>: {
                <variable>: {
                  "annual": {mean, std, ...},
                  "monthly": {1: {...}, 2: {...}, ..., 12: {...}}
                }
              }
            }
          }
        }
    """
    priors: dict = {}

    # Anual: cidade → período → variável → annual stats
    for _, row in annual.iterrows():
        city = row["city"]
        period = row["period"]
        var = row["variable"]
        priors.setdefault(city, {}).setdefault(period, {}).setdefault(var, {})
        priors[city][period][var]["annual"] = {
            k: (None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else float(v)))
            for k, v in row.items() if k not in ("city", "variable", "period")
        }

    # Mensal: cidade → período → variável → monthly[mês] → stats
    for _, row in monthly.iterrows():
        city = row["city"]
        period = row["period"]
        var = row["variable"]
        month = int(row["month"])
        priors.setdefault(city, {}).setdefault(period, {}).setdefault(var, {})
        priors[city][period][var].setdefault("monthly", {})
        priors[city][period][var]["monthly"][str(month)] = {
            k: (None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else float(v)))
            for k, v in row.items() if k not in ("city", "variable", "period", "month")
        }

    payload = {
        "metadata": {
            "title": "Climatological priors for Bayesian water balance framework",
            "wmo_periods": list(WMO_PERIODS.keys()),
            "variables": list(AGG_RULES.keys()),
            "variable_units": VAR_UNITS,
            "aggregation_rules": AGG_RULES,
            "cities": CITIES,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        },
        "priors": priors,
    }

    out = OUT_DIR / "climatological_priors.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    size_kb = out.stat().st_size / 1024
    print(f"  JSON     : {out} ({size_kb:.0f} KB, hierárquico, pronto p/ PyMC)")


def save_xlsx(annual: pd.DataFrame, monthly: pd.DataFrame, tests: pd.DataFrame) -> None:
    """XLSX para conveniência do orientador/banca (mais lento, opcional)."""
    xlsx_dir = OUT_DIR / "xlsx"
    xlsx_dir.mkdir(parents=True, exist_ok=True)

    try:
        path_annual = xlsx_dir / "normals_annual.xlsx"
        with pd.ExcelWriter(path_annual, engine="openpyxl") as writer:
            annual.to_excel(writer, sheet_name="all_data", index=False, float_format="%.3f")
            for period in WMO_PERIODS:
                sub = annual[annual["period"] == period].copy()
                sub_pivot = sub.pivot(index="city", columns="variable",
                                       values=["mean", "std", "cv"])
                sub_pivot.to_excel(writer, sheet_name=f"period_{period}",
                                   float_format="%.2f")

        path_monthly = xlsx_dir / "normals_monthly.xlsx"
        with pd.ExcelWriter(path_monthly, engine="openpyxl") as writer:
            monthly.to_excel(writer, sheet_name="all_data", index=False, float_format="%.3f")
            for var in AGG_RULES:
                sub = monthly[monthly["variable"] == var].copy()
                pivot = sub.pivot_table(
                    index=["city", "period"], columns="month", values="mean")
                pivot.to_excel(writer, sheet_name=f"var_{var}", float_format="%.2f")

        path_tests = xlsx_dir / "period_comparison_tests.xlsx"
        with pd.ExcelWriter(path_tests, engine="openpyxl") as writer:
            tests.to_excel(writer, sheet_name="kruskal_and_pairwise",
                           index=False, float_format="%.4f")
        print(f"  XLSX     : {xlsx_dir} (3 arquivos, para revisão humana)")
    except ImportError:
        print(f"  XLSX     : PULADO (instale com: pip install openpyxl)")


# ---------------------------------------------------------------------------
# Saída — LaTeX
# ---------------------------------------------------------------------------


def make_latex_annual_summary(annual: pd.DataFrame) -> str:
    """Tabela LaTeX: cidade × variável × período (mean ± std)."""
    pivot = annual.pivot(index="city", columns=["variable", "period"], values="mean")
    pivot_std = annual.pivot(index="city", columns=["variable", "period"], values="std")

    # Foco nas variáveis principais para o paper
    main_vars = ["pr", "ETo", "Tmax", "Tmin"]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Climatological normals for the 10 MATOPIBA hubs across the "
                 r"three WMO standard reference periods. Values are mean $\pm$ standard "
                 r"deviation of annual aggregates.}")
    lines.append(r"\label{tab:climatological_normals}")
    lines.append(r"\footnotesize")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")

    n_cols = 1 + len(main_vars) * len(WMO_PERIODS)
    col_spec = "l" + "c" * (n_cols - 1)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Cabeçalho 1: variáveis
    header1 = ["\\textbf{City}"]
    for var in main_vars:
        header1.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{VAR_LABELS[var]} ({VAR_UNITS[var]})}}}}")
    lines.append(" & ".join(header1) + r" \\")

    # Cabeçalho 2: períodos
    header2 = [""]
    for _ in main_vars:
        for period in WMO_PERIODS:
            header2.append(f"\\textbf{{{period}}}")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    # Linhas
    for city in CITIES:
        if city not in pivot.index:
            continue
        row = [city.replace("_", " ")]
        for var in main_vars:
            for period in WMO_PERIODS:
                try:
                    m = pivot.loc[city, (var, period)]
                    s = pivot_std.loc[city, (var, period)]
                    row.append(f"{m:.1f} $\\pm$ {s:.1f}")
                except KeyError:
                    row.append("--")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_latex_period_comparison(tests: pd.DataFrame) -> str:
    """Tabela LaTeX: testes Kruskal-Wallis + pairwise por cidade × variável."""
    main_vars = ["pr", "ETo", "Tmax", "Tmin"]

    def sig_marker(p: float) -> str:
        if pd.isna(p):
            return "--"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    use_bonferroni = "kruskal_p_bonferroni" in tests.columns
    p_col = "kruskal_p_bonferroni" if use_bonferroni else "kruskal_p_raw"
    correction_note = (
        "p-values are Bonferroni-corrected for multiple testing across "
        "all city $\\times$ variable combinations" if use_bonferroni else
        "p-values are reported uncorrected"
    )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical tests for differences in annual climate "
                 r"between WMO reference periods. K-W: Kruskal-Wallis test among "
                 r"the three periods; $\Delta_{med}$(P3-P1) is the median "
                 r"difference between the most recent (1991-2020) and the "
                 r"earliest (1961-1990) reference periods. Significance: "
                 r"*** $p<0.001$, ** $p<0.01$, * $p<0.05$, ns: not significant. "
                 + correction_note + ".}")
    lines.append(r"\label{tab:period_comparison_tests}")
    lines.append(r"\footnotesize")

    col_spec = "l" + "c" * (len(main_vars) * 2)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header1 = ["\\textbf{City}"]
    for var in main_vars:
        header1.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{var}}}}}")
    lines.append(" & ".join(header1) + r" \\")

    header2 = [""]
    for _ in main_vars:
        header2 += [r"\textbf{K-W $p$}", r"\textbf{$\Delta_{med}$(P3-P1)}"]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for city in CITIES:
        sub = tests[tests["city"] == city]
        if sub.empty:
            continue
        row = [city.replace("_", " ")]
        for var in main_vars:
            row_var = sub[sub["variable"] == var]
            if row_var.empty:
                row += ["--", "--"]
                continue
            r0 = row_var.iloc[0]
            row.append(sig_marker(r0[p_col]))
            diff = r0["delta_median_p3_minus_p1"]
            row.append(f"{diff:+.2f}" if not pd.isna(diff) else "--")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_latex_tables(annual: pd.DataFrame, tests: pd.DataFrame) -> None:
    """Gera arquivos .tex prontos para colar no paper."""
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    table1 = LATEX_DIR / "table_annual_normals.tex"
    table1.write_text(make_latex_annual_summary(annual), encoding="utf-8")
    print(f"  LaTeX 1  : {table1}")

    table2 = LATEX_DIR / "table_period_comparison.tex"
    table2.write_text(make_latex_period_comparison(tests), encoding="utf-8")
    print(f"  LaTeX 2  : {table2}")


# ---------------------------------------------------------------------------
# Resumo no console
# ---------------------------------------------------------------------------


def print_summary(annual: pd.DataFrame, tests: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(" Resumo das normais climatológicas")
    print("=" * 78)

    for var in ["pr", "ETo", "Tmax"]:
        print(f"\n  {VAR_LABELS[var]} (média entre as 10 cidades, anual):")
        for period in WMO_PERIODS:
            sub = annual[(annual["variable"] == var) & (annual["period"] == period)]
            mean_of_means = sub["mean"].mean()
            std_of_means = sub["mean"].std()
            print(f"    {period}:  {mean_of_means:7.1f} ± {std_of_means:.1f} {VAR_UNITS[var]}")

    p_col = "kruskal_p_bonferroni" if "kruskal_p_bonferroni" in tests.columns else "kruskal_p_raw"
    print(f"\n  Testes Kruskal-Wallis significativos (p<0.05, "
          f"{'Bonferroni-corrigido' if 'bonferroni' in p_col else 'p brutos'}):")
    sig = tests[tests[p_col] < 0.05]
    if sig.empty:
        print("    Nenhum teste detectou diferença significativa entre os 3 períodos.")
    else:
        for var in AGG_RULES:
            sig_var = sig[sig["variable"] == var]
            if not sig_var.empty:
                cities_sig = sig_var["city"].tolist()
                print(f"    {var:5s}: {len(cities_sig):2d}/10 cidades — "
                      f"{', '.join(c.replace('_', ' ') for c in cities_sig[:3])}"
                      f"{'...' if len(cities_sig) > 3 else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 78)
    print(" Normais Climatológicas WMO para os 10 polos MATOPIBA")
    print("=" * 78)
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Output dir    : {OUT_DIR}")
    print(f"Períodos OMM  : {list(WMO_PERIODS.keys())}")
    print(f"Variáveis     : {list(AGG_RULES.keys())}")
    print()

    # 1. Carregar todas as cidades
    print("Carregando CSVs ...")
    data = {}
    for city in CITIES:
        try:
            data[city] = load_city(city)
        except FileNotFoundError as e:
            print(f"  AVISO: {city} pulada — {e}")

    if not data:
        print("Nenhum dado carregado. Verifique DATA_DIR.")
        return

    print(f"  {len(data)} cidades carregadas.\n")

    # 2. Estatísticas anuais
    print("Calculando estatísticas anuais ...")
    annual = compute_annual_normals_table(data)
    print(f"  {len(annual)} linhas (cidade × variável × período).")

    # 3. Estatísticas mensais
    print("Calculando climatologia mensal ...")
    monthly = compute_monthly_normals_table(data)
    print(f"  {len(monthly)} linhas (cidade × variável × período × mês).")

    # 4. Testes entre períodos
    print("Aplicando testes Kruskal-Wallis e Mann-Whitney pairwise ...")
    tests = compute_period_comparison_tests(data)
    print(f"  {len(tests)} testes realizados.")

    # 5. Salvar — múltiplos formatos para usos distintos
    print("\nSalvando arquivos ...")
    save_parquet(annual, monthly, tests)        # pipeline do modelo (mais rápido)
    save_json_priors(annual, monthly)            # priors estruturados p/ PyMC
    save_csv(annual, monthly, tests)             # inspeção humana, git diff
    save_xlsx(annual, monthly, tests)            # revisão pela orientadora/banca
    save_latex_tables(annual, tests)             # colar no paper

    # 6. Resumo
    print_summary(annual, tests)

    print("\nConcluído.")


if __name__ == "__main__":
    main()