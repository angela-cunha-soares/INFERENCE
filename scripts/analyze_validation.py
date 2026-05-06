"""Analyse the 150-combination validation grid and produce paper artefacts.

Reads a ``validation_<TS>.csv`` produced by ``bwb validate`` and writes:

* PNG figures into ``figures/paper/``
* LaTeX tables into ``output/paper_tables/``
* CSV summaries into ``output/paper_tables/``

Examples
--------
python scripts/analyze_validation.py
python scripts/analyze_validation.py --csv output/validation/validation_<TS>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _set_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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


def _latest_csv(base: Path) -> Path:
    candidates = sorted(base.glob("validation_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No validation CSV found in {base}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Tables (CSV + LaTeX)
# ---------------------------------------------------------------------------


_METRIC_LABELS = {
    "metric_rmse": "RMSE",
    "metric_mae": "MAE",
    "metric_bias": "Bias",
    "metric_pbias": "PBias (\\%)",
    "metric_kge": "KGE",
    "metric_nse": "NSE",
    "metric_crps_mean": "CRPS",
    "metric_pit_alpha": r"$\alpha$-PIT",
    "metric_coverage_90": "Coverage$_{90\\%}$",
    "metric_interval_score_90": "IS$_{90\\%}$",
}


def write_global_summary(df: pd.DataFrame, out_dir: Path) -> dict:
    rows = []
    for col, label in _METRIC_LABELS.items():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "metric": label,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
        })
    summary = pd.DataFrame(rows)
    csv_path = out_dir / "table_global_metrics.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")

    # LaTeX
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Metric & Mean & Median & Std & Min & Max \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['metric']} & {row['mean']:.3f} & {row['median']:.3f} & "
            f"{row['std']:.3f} & {row['min']:.3f} & {row['max']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path = out_dir / "table_global_metrics.tex"
    tex_path.write_text("\n".join(lines))
    return {"csv": str(csv_path), "tex": str(tex_path)}


def write_per_city_summary(df: pd.DataFrame, out_dir: Path) -> dict:
    g = df.groupby("city")
    summary = g.agg(
        n=("metric_kge", "size"),
        kge_mean=("metric_kge", "mean"),
        kge_min=("metric_kge", "min"),
        nse_mean=("metric_nse", "mean"),
        rmse_mean=("metric_rmse", "mean"),
        crps_mean=("metric_crps_mean", "mean"),
        coverage_mean=("metric_coverage_90", "mean"),
    ).round(4)
    csv_path = out_dir / "table_per_city.csv"
    summary.to_csv(csv_path)
    return {"csv": str(csv_path)}


def write_per_cycle_summary(df: pd.DataFrame, out_dir: Path) -> dict:
    g = df.groupby("season")
    summary = g.agg(
        n=("metric_kge", "size"),
        kge_mean=("metric_kge", "mean"),
        nse_mean=("metric_nse", "mean"),
        coverage_mean=("metric_coverage_90", "mean"),
        crps_mean=("metric_crps_mean", "mean"),
    ).round(4)
    csv_path = out_dir / "table_per_cycle.csv"
    summary.to_csv(csv_path)
    return {"csv": str(csv_path)}


def write_per_depth_summary(df: pd.DataFrame, out_dir: Path) -> dict:
    g = df.groupby("soil_depth_cm")
    summary = g.agg(
        n=("metric_kge", "size"),
        kge_mean=("metric_kge", "mean"),
        nse_mean=("metric_nse", "mean"),
        coverage_mean=("metric_coverage_90", "mean"),
    ).round(4)
    csv_path = out_dir / "table_per_depth.csv"
    summary.to_csv(csv_path)
    return {"csv": str(csv_path)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_heatmap_kge(plt, df: pd.DataFrame, fig_dir: Path) -> str:
    """KGE heatmap city x cycle (averaged across depths)."""
    pivot = df.pivot_table(
        index="city", columns="season", values="metric_kge", aggfunc="mean",
    ).round(3)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.78, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([c.replace("_", " ") for c in pivot.index])
    ax.set_title("KGE by city x cycle (avg over soil depth)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("KGE")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            colour = "white" if v < 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=colour)
    fig.tight_layout()
    out = fig_dir / "fig_validation_kge_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def fig_metric_boxplots(plt, df: pd.DataFrame, fig_dir: Path) -> list[str]:
    """Box plots: distribution of each metric across the 150 combos, per city."""
    metrics = [
        ("metric_kge", "KGE", (0.75, 1.0)),
        ("metric_nse", "NSE", (0.94, 1.0)),
        ("metric_coverage_90", r"Coverage$_{90}$", (0.84, 0.99)),
        ("metric_crps_mean", "CRPS", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    cities = sorted(df["city"].unique())
    cities_label = [c.replace("_", " ") for c in cities]
    for ax, (col, label, ylim) in zip(axes.ravel(), metrics):
        data = [df[df["city"] == c][col].to_numpy() for c in cities]
        bp = ax.boxplot(data, tick_labels=cities_label, patch_artist=True,
                        medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_alpha(0.5)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Distribution of validation metrics per city (n=15 each)")
    fig.tight_layout()
    out = fig_dir / "fig_validation_metric_boxplots.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return [str(out)]


def fig_coverage_vs_nominal(plt, df: pd.DataFrame, fig_dir: Path) -> str:
    """Reliability diagram: empirical coverage vs nominal 0.90 across all combos."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cov = df["metric_coverage_90"].dropna().to_numpy()
    ax.hist(cov, bins=20, color="#1f77b4", alpha=0.7, edgecolor="black")
    ax.axvline(0.90, color="red", linestyle="--", lw=1.5, label="nominal 0.90")
    ax.axvline(cov.mean(), color="black", linestyle=":", lw=1.2,
               label=f"empirical mean {cov.mean():.3f}")
    ax.set_xlabel(r"Empirical coverage at the 90% credible interval")
    ax.set_ylabel("Number of combinations")
    ax.set_title("Coverage_90 across the 150-combination grid")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = fig_dir / "fig_validation_coverage_histogram.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def fig_kge_vs_crps(plt, df: pd.DataFrame, fig_dir: Path) -> str:
    """Scatter: deterministic skill (KGE) vs probabilistic skill (CRPS)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cities = sorted(df["city"].unique())
    cmap = plt.get_cmap("tab10")
    for i, city in enumerate(cities):
        sub = df[df["city"] == city]
        ax.scatter(sub["metric_kge"], sub["metric_crps_mean"],
                   s=20, alpha=0.7, label=city.replace("_", " "),
                   color=cmap(i % 10))
    ax.set_xlabel("KGE")
    ax.set_ylabel("CRPS")
    ax.set_title("Deterministic vs probabilistic skill")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    out = fig_dir / "fig_validation_kge_vs_crps.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def fig_pit_alpha_per_city(plt, df: pd.DataFrame, fig_dir: Path) -> str:
    """Bar chart of mean alpha-PIT per city (calibration check)."""
    g = df.groupby("city")["metric_pit_alpha"].agg(["mean", "std"])
    g = g.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"], color="#2ca02c", alpha=0.8,
           edgecolor="black", capsize=3)
    ax.axhline(1.0, color="red", linestyle="--", lw=1, label="perfect calibration")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in g.index], rotation=45,
                       ha="right")
    ax.set_ylabel(r"$\alpha$-PIT (Renard et al. 2010)")
    ax.set_title("PIT calibration per city (mean +/- 1 std)")
    ax.set_ylim(0.78, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = fig_dir / "fig_validation_pit_alpha_per_city.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=None,
                   help="Path to validation CSV (default: most recent in output/validation/)")
    p.add_argument("--fig-dir", type=Path, default=Path("figures/paper"))
    p.add_argument("--table-dir", type=Path, default=Path("output/paper_tables"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    csv = args.csv or _latest_csv(Path("output/validation"))
    print(f"Reading {csv}")
    df = pd.read_csv(csv)
    df = df[df["status"] == "ok"].copy()
    print(f"  {len(df)} successful runs")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    plt = _set_style()

    artefacts = {"tables": [], "figures": []}

    print("\nWriting tables...")
    artefacts["tables"].append(write_global_summary(df, args.table_dir))
    artefacts["tables"].append(write_per_city_summary(df, args.table_dir))
    artefacts["tables"].append(write_per_cycle_summary(df, args.table_dir))
    artefacts["tables"].append(write_per_depth_summary(df, args.table_dir))
    for t in artefacts["tables"]:
        for k, v in t.items():
            print(f"  [{k}] {v}")

    print("\nWriting figures...")
    artefacts["figures"].append(fig_heatmap_kge(plt, df, args.fig_dir))
    artefacts["figures"].extend(fig_metric_boxplots(plt, df, args.fig_dir))
    artefacts["figures"].append(fig_coverage_vs_nominal(plt, df, args.fig_dir))
    artefacts["figures"].append(fig_kge_vs_crps(plt, df, args.fig_dir))
    artefacts["figures"].append(fig_pit_alpha_per_city(plt, df, args.fig_dir))
    for f in artefacts["figures"]:
        print(f"  [png] {f}")

    print(f"\nDone. {len(artefacts['figures'])} figures, "
          f"{len(artefacts['tables'])} table sets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
