"""Generate publication-ready figures for the manuscript.

Reads existing results in ``output/`` (deterministic baselines, posterior
forecasts, recommended irrigation depths) and produces a curated set of
PNG/PDF figures into ``figures/paper/``.

Examples
--------
python scripts/generate_paper_figures.py
python scripts/generate_paper_figures.py --output figures/paper --no-pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CYCLES = ["2020_2021", "2021_2022", "2022_2023", "2023_2024", "2024_2025"]


def _set_style() -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _save(fig, base: Path, save_pdf: bool) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [str(base.with_suffix(".png"))]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    if save_pdf:
        paths.append(str(base.with_suffix(".pdf")))
        fig.savefig(paths[1], bbox_inches="tight")
    return paths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_water_balance_timeseries(
    plt,
    output_dir: Path,
    figures_dir: Path,
    save_pdf: bool,
) -> list[str]:
    """Posterior soil-moisture trajectories for every cycle (one panel each)."""
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(11, 6), sharey=True)
    axes = axes.ravel()

    saved = []
    for i, cycle in enumerate(CYCLES):
        ax = axes[i]
        prev_path = output_dir / f"previsao_{cycle}.csv"
        det_path = output_dir / f"deterministico_{cycle}.csv"
        if not prev_path.exists() or not det_path.exists():
            ax.set_visible(False)
            continue
        prev = pd.read_csv(prev_path)
        det = pd.read_csv(det_path)
        d = prev["dia_ciclo"].to_numpy()

        ax.fill_between(d, prev["SW_q05"], prev["SW_q95"],
                        color="#1f77b4", alpha=0.25, label="Bayes 90% CI")
        ax.plot(d, prev["SW_media"], color="#1f77b4", lw=1.4, label="Bayes mean")
        if "SW" in det.columns:
            ax.plot(det["dia_ciclo"], det["SW"], color="#d62728", lw=1.0,
                    ls="--", label="FAO-56 baseline")
        ax.set_title(cycle.replace("_", "/"))
        ax.set_xlabel("Dia do ciclo")
        if i % cols == 0:
            ax.set_ylabel(r"$\theta$ / SW (mm)")
        if i == 0:
            ax.legend(loc="lower left", frameon=False, fontsize=8)

    for j in range(len(CYCLES), rows * cols):
        axes[j].set_visible(False)

    fig.suptitle("Posterior soil-water trajectories — Balsas, MA (2020–2025)",
                 fontsize=11)
    fig.tight_layout()
    saved += _save(fig, figures_dir / "fig_water_balance_timeseries", save_pdf)
    plt.close(fig)
    return saved


def fig_irrigation_summary(
    plt,
    output_dir: Path,
    figures_dir: Path,
    save_pdf: bool,
) -> list[str]:
    """Recommended irrigation depths and probability of irrigation per cycle."""
    laminas_path = output_dir / "laminas_recomendadas.csv"
    if not laminas_path.exists():
        return []
    laminas = pd.read_csv(laminas_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    cycles = laminas["Ciclo"].astype(str).to_numpy()
    x = np.arange(len(cycles))

    ax1.bar(x, laminas["Prob_irrigar_%"], color="#1f77b4", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cycles, rotation=20)
    ax1.set_ylabel("P(irrigar) [%]")
    ax1.set_title("Probabilidade posterior de irrigação")

    ax2.errorbar(
        x, laminas["I_total_media"],
        yerr=[laminas["I_total_media"] - laminas["I_total_q05"],
              laminas["I_total_q95"] - laminas["I_total_media"]],
        fmt="o", color="#2ca02c", capsize=4, lw=1.2,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(cycles, rotation=20)
    ax2.set_ylabel("Lâmina total (mm)")
    ax2.set_title("Lâmina recomendada por ciclo (90% CI)")

    fig.tight_layout()
    paths = _save(fig, figures_dir / "fig_irrigation_summary", save_pdf)
    plt.close(fig)
    return paths


def fig_metrics_summary(
    plt,
    output_dir: Path,
    figures_dir: Path,
    save_pdf: bool,
) -> list[str]:
    """Probabilistic metrics: coverage of 90% CI vs. deterministic percentile."""
    metrics_path = output_dir / "metricas_probabilisticas.csv"
    if not metrics_path.exists():
        return []
    df = pd.read_csv(metrics_path)
    if "Ciclo" not in df.columns:
        return []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    cycles = df["Ciclo"].astype(str).to_numpy()
    x = np.arange(len(cycles))

    ax1.bar(x, df["Percentil_det"], color="#9467bd", alpha=0.85)
    ax1.axhline(50, color="k", ls=":", lw=0.8, label="mediana esperada")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cycles, rotation=20)
    ax1.set_ylabel("Percentil do FAO-56 dentro da posterior")
    ax1.set_title("Calibração ponto-a-ponto")
    ax1.legend(frameon=False)

    ax2.bar(x, df["CRPS"], color="#ff7f0e", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cycles, rotation=20)
    ax2.set_ylabel("CRPS (mm)")
    ax2.set_title("CRPS — quanto menor, melhor")

    fig.tight_layout()
    paths = _save(fig, figures_dir / "fig_metrics_probabilisticas", save_pdf)
    plt.close(fig)
    return paths


def fig_kc_curve(plt, figures_dir: Path, save_pdf: bool) -> list[str]:
    """FAO-56 Kc curve used in this study (90-day soybean)."""
    from bwb.phenology.kc_curves import soybean_kc_90d

    kc = soybean_kc_90d()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(np.arange(1, len(kc) + 1), kc, color="#2ca02c", lw=1.6)
    ax.fill_between(np.arange(1, len(kc) + 1), 0, kc, color="#2ca02c", alpha=0.12)
    for boundary, label in zip([15, 30, 70, 90], ["ini", "dev", "mid", "late"]):
        ax.axvline(boundary, color="grey", ls=":", lw=0.7)
        ax.text(boundary - 1, 1.18, label, ha="right", color="grey", fontsize=8)
    ax.set_xlim(0, len(kc) + 1)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Dia do ciclo")
    ax.set_ylabel(r"$K_c$")
    ax.set_title("FAO-56 Kc — Soja 90-dias (early)")
    fig.tight_layout()
    paths = _save(fig, figures_dir / "fig_kc_curve_soybean_90d", save_pdf)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="figures/paper",
                   help="Where to write the figures (default: figures/paper)")
    p.add_argument("--results", default="output",
                   help="Directory containing model outputs (default: output)")
    p.add_argument("--no-pdf", action="store_true", help="Skip PDF copies")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    plt = _set_style()

    output_dir = Path(args.results)
    figures_dir = Path(args.output)

    print("=" * 72)
    print(" Generating paper figures")
    print("=" * 72)
    print(f" Results dir : {output_dir}")
    print(f" Figures dir : {figures_dir}")
    print()

    saved = []
    saved += fig_water_balance_timeseries(plt, output_dir, figures_dir, not args.no_pdf)
    saved += fig_irrigation_summary(plt, output_dir, figures_dir, not args.no_pdf)
    saved += fig_metrics_summary(plt, output_dir, figures_dir, not args.no_pdf)
    saved += fig_kc_curve(plt, figures_dir, not args.no_pdf)

    if not saved:
        print("No figures generated — check that output/ contains the model results.")
        return 1

    for p in saved:
        print(f"  saved  {p}")
    print()
    print(f"Total figures: {len(saved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
