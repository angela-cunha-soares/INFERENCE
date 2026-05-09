"""Generate climatological-forecast figures for the manuscript (English).

Reads the May 2026 climatological sequential forecast outputs in
``output/forecast_sequential/`` and produces three publication-ready
figures into ``figures/paper/``:

  * ``fig_forecast_calibration.png`` -- realised vs forecast median scatter
    of seasonal irrigation depth, 50 city x cycle combinations, with
    central 90% credible-interval error bars.
  * ``fig_forecast_intervals_by_cycle.png`` -- per-cycle credible intervals
    of seasonal irrigation depth across the 10 hubs, with realised values
    overlaid.
  * ``fig_forecast_trajectory_balsas.png`` -- daily soil-water trajectory
    at Balsas across the five validation cycles, showing the climatological
    forecast 90% band against the deterministic baseline.

All labels and titles are in English.

Examples
--------
python scripts/generate_forecast_seq_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CYCLES = ["2020_2021", "2021_2022", "2022_2023", "2023_2024", "2024_2025"]
CYCLE_LABELS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]

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

CITY_DISPLAY = {
    "Baixa_Grande_do_Ribeiro": "Baixa Grande do Ribeiro",
    "Balsas": "Balsas",
    "Barreiras": "Barreiras",
    "Bom_Jesus": "Bom Jesus",
    "Campos_Lindos": "Campos Lindos",
    "Correntina": "Correntina",
    "Formosa_do_Rio_Preto": "Formosa do Rio Preto",
    "Luis_Eduardo_Magalhaes": "Luis Eduardo Magalhaes",
    "Tasso_Fragoso": "Tasso Fragoso",
    "Urucui": "Urucui",
}

CYCLE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _set_style() -> None:
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


def _save(fig, base: Path) -> str:
    base.parent.mkdir(parents=True, exist_ok=True)
    out = base.with_suffix(".png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return str(out)


def fig_forecast_calibration(
    summary: pd.DataFrame,
    figures_dir: Path,
) -> str:
    """Realised vs forecast q50, with 90% CI error bars, 50 points."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    obs = summary["det_I_total_obs_mm"].to_numpy()
    q05 = summary["prob_I_total_q05"].to_numpy()
    q50 = summary["det_I_total_forecast_median_mm"].to_numpy()
    q95 = summary["prob_I_total_q95"].to_numpy()
    cycles = summary["cycle"].to_numpy()

    cycle_to_color = dict(zip([c.replace("_", "/") for c in CYCLES], CYCLE_COLORS))

    for cyc, label, color in zip(
        [c.replace("_", "/") for c in CYCLES], CYCLE_LABELS, CYCLE_COLORS,
    ):
        mask = cycles == cyc
        ax.errorbar(
            q50[mask], obs[mask],
            xerr=[q50[mask] - q05[mask], q95[mask] - q50[mask]],
            fmt="o", color=color, ecolor=color, alpha=0.7,
            capsize=2.5, lw=1.0, ms=5, label=label,
        )

    upper = max(obs.max(), q95.max()) * 1.05
    ax.plot([0, upper], [0, upper], color="black", lw=0.9, ls="--", alpha=0.6,
            label="1:1 (oracle)")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Forecast median $I_{total}$ (mm)")
    ax.set_ylabel("Realised $I_{total}$ from FAO-56 baseline (mm)")
    ax.set_title("Climatological forecast calibration on seasonal irrigation depth\n"
                 "50 city $\\times$ cycle combinations, MATOPIBA 2020/21--2024/25")
    ax.legend(title="Cycle", loc="upper left", frameon=False, ncol=2)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = _save(fig, figures_dir / "fig_forecast_calibration")
    plt.close(fig)
    return path


def fig_forecast_intervals_by_cycle(
    summary: pd.DataFrame,
    figures_dir: Path,
) -> str:
    """Per-cycle credible intervals of I_total across the 10 hubs."""
    fig, axes = plt.subplots(5, 1, figsize=(9.0, 11.0), sharex=True)

    city_disp = [CITY_DISPLAY[c] for c in CITIES]
    x = np.arange(len(CITIES))

    for ax, cycle_key, cycle_label, color in zip(
        axes, [c.replace("_", "/") for c in CYCLES], CYCLE_LABELS, CYCLE_COLORS,
    ):
        sub = summary[summary["cycle"] == cycle_key].set_index("city")
        sub = sub.reindex(CITIES)

        q05 = sub["prob_I_total_q05"].to_numpy()
        q50 = sub["det_I_total_forecast_median_mm"].to_numpy()
        q95 = sub["prob_I_total_q95"].to_numpy()
        obs = sub["det_I_total_obs_mm"].to_numpy()

        ax.errorbar(
            x, q50,
            yerr=[q50 - q05, q95 - q50],
            fmt="o", color=color, ecolor=color, alpha=0.85,
            capsize=4, lw=1.2, ms=6, label="Forecast 90% CI",
        )
        ax.scatter(x, obs, marker="*", s=85, color="black", zorder=5,
                   label="Realised")

        ax.set_ylabel("$I_{total}$ (mm)")
        ax.set_title(f"Cycle {cycle_label}", loc="left")
        ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
        if ax is axes[0]:
            ax.legend(loc="upper right", frameon=False, ncol=2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(city_disp, rotation=35, ha="right")

    fig.suptitle(
        "Climatological forecast 90% credible intervals on seasonal irrigation depth",
        y=0.995, fontsize=11,
    )
    fig.tight_layout()
    path = _save(fig, figures_dir / "fig_forecast_intervals_by_cycle")
    plt.close(fig)
    return path


def fig_forecast_trajectory_balsas(
    forecast_dir: Path,
    figures_dir: Path,
) -> str:
    """Daily SW trajectory at Balsas across the five validation cycles."""
    fig, axes = plt.subplots(5, 1, figsize=(9.0, 11.0), sharex=True)

    for ax, cycle, cycle_label, color in zip(
        axes, CYCLES, CYCLE_LABELS, CYCLE_COLORS,
    ):
        forecast = pd.read_csv(forecast_dir / f"forecast_Balsas_{cycle}.csv")
        observed = pd.read_csv(forecast_dir / f"observed_Balsas_{cycle}.csv")

        d = forecast["dia_ciclo"].to_numpy()
        ax.fill_between(
            d, forecast["SW_q05"], forecast["SW_q95"],
            color=color, alpha=0.20, label="Forecast 90% CI",
        )
        ax.plot(d, forecast["SW_q50"], color=color, lw=1.4, ls="-",
                label="Forecast median")
        ax.plot(observed["dia_ciclo"], observed["SW_obs"],
                color="black", lw=1.1, ls="--", label="FAO-56 baseline")

        ax.set_ylabel("SW (mm)")
        ax.set_title(f"Cycle {cycle_label}", loc="left")
        ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
        if ax is axes[0]:
            ax.legend(loc="lower left", frameon=False, ncol=3)

    axes[-1].set_xlabel("Day of cycle")
    fig.suptitle(
        "Daily soil-water content at Balsas (MA): climatological forecast vs FAO-56 baseline",
        y=0.995, fontsize=11,
    )
    fig.tight_layout()
    path = _save(fig, figures_dir / "fig_forecast_trajectory_balsas")
    plt.close(fig)
    return path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecast-dir", default="output/forecast_sequential",
                   help="Directory with sequential-forecast outputs")
    p.add_argument("--figures-dir", default="figures/paper",
                   help="Directory to write figures")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _set_style()

    forecast_dir = Path(args.forecast_dir)
    figures_dir = Path(args.figures_dir)
    summary = pd.read_csv(forecast_dir / "sequential_summary_all_cities.csv")

    print("=" * 72)
    print(" Generating forecast-sequential figures (English)")
    print("=" * 72)
    print(f" Forecast dir : {forecast_dir}")
    print(f" Figures dir  : {figures_dir}")
    print(f" Combinations : {len(summary)}")
    print()

    saved = [
        fig_forecast_calibration(summary, figures_dir),
        fig_forecast_intervals_by_cycle(summary, figures_dir),
        fig_forecast_trajectory_balsas(forecast_dir, figures_dir),
    ]
    for p in saved:
        print(f"  saved  {p}")
    print(f"\nTotal figures: {len(saved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
