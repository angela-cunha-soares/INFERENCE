"""Compare reduced-input ETo formulas vs the Fused FAO-56 Penman-Monteith.

Demonstrates the value of the multi-source fused product by showing
that two widely-used reduced-variable alternatives (Hargreaves-Samani
and Benavides-Lopez) drift much further from the gold-standard
Penman-Monteith than the fused product does, on identical data and
period.

Outputs
-------
* ``output/paper_tables/table_eto_alternatives.csv``  — RMSE, bias, r
  per formula, per city, vs the Penman-Monteith reference.
* ``output/paper_tables/table_eto_alternatives.tex`` — paper-ready table.
* ``figures/paper/fig_eto_alternatives.png`` — scatter / time-series
  comparison for a representative city.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bwb.data.loaders import load_city_series
from bwb.data.sources.climate import (
    compute_eto_benavides_lopez,
    compute_eto_hargreaves_samani,
)

CITIES = {
    "Baixa_Grande_do_Ribeiro": (-8.33, -45.09, 519.0),
    "Balsas":                  (-7.53, -46.04, 263.0),
    "Barreiras":               (-12.12, -45.03, 474.0),
    "Bom_Jesus":               (-9.07, -44.36, 277.0),
    "Campos_Lindos":           (-7.99, -46.86, 540.0),
    "Correntina":              (-13.33, -44.62, 552.0),
    "Formosa_do_Rio_Preto":    (-11.04, -45.20, 489.0),
    "Luis_Eduardo_Magalhaes":  (-12.08, -45.71, 748.0),
    "Tasso_Fragoso":           (-8.47, -45.75, 280.0),
    "Urucui":                  (-7.44, -44.34, 399.0),
}

OUT_TABLES = ROOT / "output" / "paper_tables"
OUT_FIGS = ROOT / "figures" / "paper"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


def metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 5:
        return {"n": int(m.sum())}
    obs, sim = obs[m], sim[m]
    bias = float(np.mean(sim - obs))
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    if obs.std() > 1e-9 and sim.std() > 1e-9:
        r = float(np.corrcoef(obs, sim)[0, 1])
    else:
        r = float("nan")
    return {"n": int(m.sum()), "bias": bias, "rmse": rmse, "r": r}


def main():
    rows = []
    for city, (lat, lon, elev) in CITIES.items():
        df = load_city_series(city)
        df["date"] = pd.to_datetime(df["date"])
        df["doy"] = df["date"].dt.dayofyear
        # Tmean if present; otherwise mean of Tmax & Tmin
        if "Tmean" not in df.columns:
            df["Tmean"] = 0.5 * (df["Tmax"] + df["Tmin"])

        eto_pm = df["ETo"].to_numpy()
        eto_hs = compute_eto_hargreaves_samani(
            df["Tmax"].to_numpy(), df["Tmin"].to_numpy(),
            doy=df["doy"].to_numpy(), lat_deg=lat,
        )
        eto_bl = compute_eto_benavides_lopez(
            df["Tmean"].to_numpy(), df["RH"].to_numpy(),
        )

        for name, eto_alt in [("Hargreaves-Samani", eto_hs),
                              ("Benavides-Lopez", eto_bl)]:
            m = metrics(eto_pm, eto_alt)
            rows.append({"city": city.replace("_", " "), "formula": name,
                         **m, "pm_mean": float(np.nanmean(eto_pm))})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_TABLES / "table_eto_alternatives.csv", index=False,
                   float_format="%.4f")
    print("=" * 80)
    print("RMSE of reduced-input ETo formulas vs FAO-56 Penman-Monteith reference")
    print("Across 10 MATOPIBA cities, 1961-2025 (n approx 23,741 days each)")
    print("=" * 80)
    pivot = summary.pivot_table(index="city", columns="formula",
                                 values=["bias", "rmse", "r"]).round(3)
    print(pivot.to_string())
    print()

    pooled = (summary.groupby("formula")
              .agg(n=("n", "sum"),
                   rmse_med=("rmse", "median"),
                   bias_med=("bias", "median"),
                   r_med=("r", "median"))
              .round(3))
    print("Pooled across 10 cities (median):")
    print(pooled.to_string())

    # LaTeX
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reduced-input ETo formulas (Hargreaves-Samani 1985; "
        r"Benavides-Lopez 1970) compared to the FAO-56 Penman-Monteith "
        r"reference \citep{Allen1998} on the Xavier reanalysis daily "
        r"series, 1961--2025, for 10 MATOPIBA cities. Cells are RMSE in "
        r"mm day$^{-1}$. Both reduced formulas drift several times "
        r"farther from PM than the daily disagreement between Xavier and "
        r"the operational fused product reported in "
        r"Table~\ref{tab:inmet-validation}; this motivates retaining the "
        r"full PM equation, computed on fused inputs, for the operational "
        r"forecast.}",
        r"\label{tab:eto-alternatives}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"City & PM mean & HS RMSE & HS Bias & BL RMSE & BL Bias \\",
        r" & (mm/d) & (mm/d) & (mm/d) & (mm/d) & (mm/d) \\",
        r"\midrule",
    ]
    cities_sorted = sorted(summary["city"].unique())
    for c in cities_sorted:
        sub = summary[summary["city"] == c].set_index("formula")
        pm_mean = sub["pm_mean"].iloc[0]
        hs = sub.loc["Hargreaves-Samani"]
        bl = sub.loc["Benavides-Lopez"]
        tex.append(
            f"{c} & {pm_mean:.2f} & "
            f"{hs['rmse']:.2f} & {hs['bias']:+.2f} & "
            f"{bl['rmse']:.2f} & {bl['bias']:+.2f} \\\\"
        )
    pooled_hs = pooled.loc["Hargreaves-Samani"]
    pooled_bl = pooled.loc["Benavides-Lopez"]
    tex += [
        r"\midrule",
        f"\\textbf{{Median (10 cities)}} & --- & "
        f"\\textbf{{{pooled_hs['rmse_med']:.2f}}} & "
        f"\\textbf{{{pooled_hs['bias_med']:+.2f}}} & "
        f"\\textbf{{{pooled_bl['rmse_med']:.2f}}} & "
        f"\\textbf{{{pooled_bl['bias_med']:+.2f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = OUT_TABLES / "table_eto_alternatives.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")

    # Figure: scatter for Balsas
    import matplotlib.pyplot as plt
    df = load_city_series("Balsas")
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["Tmean"] = 0.5 * (df["Tmax"] + df["Tmin"])
    df = df[(df["date"] >= "2020-01-01") & (df["date"] < "2021-01-01")]
    eto_pm = df["ETo"].to_numpy()
    eto_hs = compute_eto_hargreaves_samani(
        df["Tmax"].to_numpy(), df["Tmin"].to_numpy(),
        doy=df["doy"].to_numpy(), lat_deg=-7.53,
    )
    eto_bl = compute_eto_benavides_lopez(
        df["Tmean"].to_numpy(), df["RH"].to_numpy(),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(df["date"], eto_pm, label="FAO-56 PM (reference)", color="black", lw=1.5)
    ax.plot(df["date"], eto_hs, label="Hargreaves-Samani", color="C1", alpha=0.8, lw=1)
    ax.plot(df["date"], eto_bl, label="Benavides-Lopez", color="C2", alpha=0.8, lw=1)
    ax.set_ylabel("ETo (mm/day)")
    ax.set_title("Balsas/MA — 2020")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.scatter(eto_pm, eto_hs, s=8, alpha=0.4, label="HS")
    ax.scatter(eto_pm, eto_bl, s=8, alpha=0.4, label="BL")
    lim = float(np.nanmax([eto_pm.max(), eto_hs.max(), eto_bl.max()]))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1")
    ax.set_xlabel("FAO-56 PM (mm/day)")
    ax.set_ylabel("Reduced formula (mm/day)")
    ax.set_title("Pointwise agreement")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = OUT_FIGS / "fig_eto_alternatives.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {tex_path.relative_to(ROOT)}")
    print(f"Wrote {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
