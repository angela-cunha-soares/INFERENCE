"""Regenerate LaTeX tables for the climatological sequential forecast.

Reads ``output/forecast_sequential/sequential_summary_all_cities.csv`` and
writes:

* ``output/paper_tables/table_forecast_seq.tex`` -- aggregate (50 combinations)
* ``output/paper_tables/table_forecast_by_city.tex``
* ``output/paper_tables/table_forecast_by_cycle.tex``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "output" / "forecast_sequential" / "sequential_summary_all_cities.csv"
OUT = ROOT / "output" / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(SUMMARY)

    # Aggregate
    metrics = {
        r"CRPS$_{I_{\mathrm{total}}}$ (mm)": "prob_CRPS_I_total_mm",
        r"Coverage$_{90,\mathrm{SW}}$": "prob_coverage_90_SW_daily",
        r"Coverage$_{90,I_{\mathrm{total}}}$": "prob_coverage_90_I_total",
        r"KGE$_{\mathrm{SW}}$": "det_KGE_SW",
        r"PIT $I_{\mathrm{total}}$": "prob_PIT_I_total",
        r"CRPSS vs.\ naive climatology": "CRPSS_vs_naive_climatology",
    }

    rows = []
    for label, col in metrics.items():
        s = df[col]
        rows.append((label, s.mean(), s.median()))

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Climatological sequential-forecast metrics across the 50 city $\times$ cycle combinations (2020/21--2024/25, 10 MATOPIBA hubs). Forecast initialised with the climatologically-informed Dirichlet prior of Section~\ref{sec:methods_forecast} ($\boldsymbol{\alpha}_{2020}$ = SPEI-tercile counts on 1961--2019 plus a uniform smoother of 1.0 per class) and updated sequentially after each observed cycle. CRPS is computed on the seasonal irrigation-depth distribution; coverage is the empirical fraction of realised values within the 90\\% credible interval (nominal: 0.90). CRPSS is the continuous-ranked-probability skill score against the naive climatological mean (positive = better than baseline).}",
        r"\label{tab:forecast_seq}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Metric & Mean & Median \\",
        r"\midrule",
    ]
    for label, mean, median in rows:
        lines.append(f"{label} & {mean:.2f} & {median:.2f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "table_forecast_seq.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per-city
    city = df.groupby("city").agg(
        CRPS=("prob_CRPS_I_total_mm", "mean"),
        Cov_SW=("prob_coverage_90_SW_daily", "mean"),
        KGE_SW=("det_KGE_SW", "mean"),
    )
    city_order = [
        "Baixa_Grande_do_Ribeiro", "Balsas", "Barreiras", "Bom_Jesus",
        "Campos_Lindos", "Correntina", "Formosa_do_Rio_Preto",
        "Luis_Eduardo_Magalhaes", "Tasso_Fragoso", "Urucui",
    ]
    city = city.reindex(city_order)
    pretty = {
        "Baixa_Grande_do_Ribeiro": "Baixa Grande do Ribeiro",
        "Balsas": "Balsas",
        "Barreiras": "Barreiras",
        "Bom_Jesus": "Bom Jesus",
        "Campos_Lindos": "Campos Lindos",
        "Correntina": "Correntina",
        "Formosa_do_Rio_Preto": "Formosa do Rio Preto",
        "Luis_Eduardo_Magalhaes": r"Lu\'is Eduardo Magalh\~aes",
        "Tasso_Fragoso": "Tasso Fragoso",
        "Urucui": r"Urucu\'i",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Mean climatological-forecast metrics by city across the five soybean cycles 2020/21--2024/25.}",
        r"\label{tab:forecast_by_city}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"City & CRPS (mm) & Coverage$_{90,\mathrm{SW}}$ & KGE$_{\mathrm{SW}}$ \\",
        r"\midrule",
    ]
    for code, row in city.iterrows():
        lines.append(
            f"{pretty[code]} & {row['CRPS']:.1f} & {row['Cov_SW']:.3f} & {row['KGE_SW']:+.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "table_forecast_by_city.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per-cycle
    cyc = df.groupby("cycle").agg(
        CRPS=("prob_CRPS_I_total_mm", "mean"),
        Cov_SW=("prob_coverage_90_SW_daily", "mean"),
        Cov_I=("prob_coverage_90_I_total", "mean"),
    )
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Mean climatological-forecast metrics by cycle across the ten MATOPIBA hubs. The 2021/22 cycle, near the second peak of the 2020--2023 ``triple-dip'' La~Ni\~na, exhibits the largest CRPS, reflecting heightened above-normal precipitation variability under that regime; the 2023/24 El~Ni\~no cycle and the 2022/23 transition cycle show the lowest CRPS as the climatologically-informed prior assigns disproportionate weight to those analogues.}",
        r"\label{tab:forecast_by_cycle}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Cycle & CRPS (mm) & Coverage$_{90,\mathrm{SW}}$ & Coverage$_{90,I_{\mathrm{total}}}$ \\",
        r"\midrule",
    ]
    for cycle, row in cyc.iterrows():
        lines.append(
            f"{cycle} & {row['CRPS']:.1f} & {row['Cov_SW']:.3f} & {row['Cov_I']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "table_forecast_by_cycle.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK: tables written to", OUT)


if __name__ == "__main__":
    main()
