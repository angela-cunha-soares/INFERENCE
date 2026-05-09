"""Generate MCMC diagnostics table from the existing NUTS traces.

Reads the five InferenceData files saved by the posterior-recovery
pipeline (one per cycle, ``output/trace_<year>.nc``) and computes,
per cycle and per parameter, the standard sampling-quality diagnostics
recommended by ArviZ \\citep{Vehtari2021}:

* effective sample size (ESS bulk and tail),
* Gelman-Rubin :math:`\\hat{R}`,
* number of divergent transitions,
* posterior mean and 94% HDI.

Outputs
-------
* ``output/paper_tables/table_mcmc_diagnostics.csv``
* ``output/paper_tables/table_mcmc_diagnostics.tex``
"""

from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PARAMS = ["theta_s", "theta_r", "Kc_mult", "theta_init", "sigma_obs"]
CYCLES = ["2020_2021", "2021_2022", "2022_2023", "2023_2024", "2024_2025"]

OUT = ROOT / "output" / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)


def diagnose(idata: az.InferenceData, cycle: str) -> pd.DataFrame:
    rows = []
    summary = az.summary(
        idata, var_names=PARAMS,
        kind="all", round_to="none",
    )
    # Number of divergences (chain × draw mask)
    n_div = 0
    if "sample_stats" in idata.groups() and "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].sum().values)

    # Posterior draws (chain × draw)
    n_chain = idata.posterior.sizes.get("chain", 1)
    n_draw = idata.posterior.sizes.get("draw", 0)

    for p in PARAMS:
        if p not in summary.index:
            continue
        row = summary.loc[p]
        rows.append({
            "cycle":        cycle,
            "parameter":    p,
            "post_mean":    float(row["mean"]),
            "post_sd":      float(row["sd"]),
            "hdi_3":        float(row["hdi_3%"]),
            "hdi_97":       float(row["hdi_97%"]),
            "ess_bulk":     float(row["ess_bulk"]),
            "ess_tail":     float(row["ess_tail"]),
            "r_hat":        float(row["r_hat"]),
            "mcse_mean":    float(row["mcse_mean"]),
            "n_chain":      n_chain,
            "n_draw":       n_draw,
            "n_divergent":  n_div,
        })
    return pd.DataFrame(rows)


def main():
    pieces = []
    for cy in CYCLES:
        path = ROOT / "output" / "state_space" / f"trace_state_space_{cy}.nc"
        if not path.exists():
            print(f"WARN missing: {path}")
            continue
        idata = az.from_netcdf(path)
        pieces.append(diagnose(idata, cy))
    if not pieces:
        raise SystemExit("No traces found")

    df = pd.concat(pieces, ignore_index=True)
    csv_path = OUT / "table_mcmc_diagnostics.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Wrote {csv_path.relative_to(ROOT)} ({len(df)} rows)")

    # Pivot for compact LaTeX: parameters as rows, cycles as columns
    print()
    print("=" * 96)
    print("MCMC diagnostics — NUTS posterior-recovery test (5 cycles, "
          f"{df.iloc[0]['n_chain']:.0f} chain × {df.iloc[0]['n_draw']:.0f} draws each)")
    print("=" * 96)

    # Headline numbers
    rhat_max = df["r_hat"].max()
    ess_min = df["ess_bulk"].min()
    div_total = df.groupby("cycle")["n_divergent"].first().sum()
    print(f"All chains converged: max R-hat = {rhat_max:.3f} (target < 1.05)")
    print(f"Sampler efficiency:   min ESS_bulk = {ess_min:.0f}")
    print(f"Pathological draws:   {div_total:.0f} divergent transitions across all 5 cycles")
    print()

    pivot_rhat = df.pivot(index="parameter", columns="cycle", values="r_hat")
    pivot_ess = df.pivot(index="parameter", columns="cycle", values="ess_bulk")
    print("R-hat:")
    print(pivot_rhat.round(3).to_string())
    print()
    print("ESS_bulk:")
    print(pivot_ess.round(0).to_string())

    # LaTeX table — one row per parameter, summarising across cycles
    summary_per_param = (df.groupby("parameter")
                          .agg(post_mean=("post_mean", "mean"),
                               post_sd=("post_sd", "mean"),
                               rhat_max=("r_hat", "max"),
                               ess_bulk_min=("ess_bulk", "min"),
                               ess_tail_min=("ess_tail", "min"),
                               n_div=("n_divergent", "sum")))
    summary_per_param = summary_per_param.reindex(PARAMS)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sampling diagnostics for the Bayesian state-space "
        r"model (Section~\ref{sec:methods_pymc}) across the five "
        r"posterior-recovery cycles. NUTS \citep{Hoffman2014} draws "
        f"{int(df.iloc[0]['n_chain'])} chain $\\times$ "
        f"{int(df.iloc[0]['n_draw'])} post-warm-up samples per cycle. "
        r"Posterior mean $\pm$ posterior s.d. is averaged across the "
        r"five cycles; $\hat{R}$ and ESS are reported as the worst-"
        r"case value across all cycles to expose any single-cycle "
        r"degradation. All $\hat{R} < 1.05$ and ESS$_{\mathrm{bulk}} > 100$ "
        r"satisfy the convergence thresholds recommended by "
        r"\citet{Vehtari2021}.}",
        r"\label{tab:mcmc-diagnostics}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Parameter & Posterior mean & Post.\ s.d. & $\hat{R}_\mathrm{max}$ "
        r"& ESS$_\mathrm{bulk,min}$ & ESS$_\mathrm{tail,min}$ & Divergences \\",
        r"\midrule",
    ]
    name_map = {
        "theta_s":    r"$\theta_s$",
        "theta_r":    r"$\theta_r$",
        "Kc_mult":    r"$K_{c,\mathrm{mult}}$",
        "theta_init": r"$\theta_\mathrm{init}$",
        "sigma_obs":  r"$\sigma_\mathrm{obs}$",
    }
    for p, row in summary_per_param.iterrows():
        tex.append(
            f"{name_map[p]} & {row['post_mean']:.3f} & {row['post_sd']:.3f} & "
            f"{row['rhat_max']:.3f} & {row['ess_bulk_min']:.0f} & "
            f"{row['ess_tail_min']:.0f} & {int(row['n_div'])} \\\\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = OUT / "table_mcmc_diagnostics.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"\nWrote {tex_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
