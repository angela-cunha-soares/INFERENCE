"""Generate paper-ready analysis figures and tables from the rolling backtest.

Reads ``output/backtest_rolling/backtest_rolling_h5d.csv`` (produced by
``bwb backtest-rolling``) and writes:

* ``figures/paper/backtest_rolling_coverage_by_city.png`` — coverage 90% CI
  per city + ensemble median bias.
* ``figures/paper/backtest_rolling_crps_distribution.png`` — CRPS histogram
  + per-city box plot.
* ``figures/paper/backtest_rolling_calibration.png`` — calibration plot
  (forecast q50 vs observed cumulative I).
* ``figures/paper/backtest_rolling_skill_by_day.png`` — CRPS and coverage
  as a function of day-of-cycle.
* ``output/paper_tables/table_backtest_rolling_summary.csv`` and ``.tex`` —
  per-city summary statistics for the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "output" / "backtest_rolling" / "backtest_rolling_h5d.csv"
PAPER_TABLES = ROOT / "output" / "paper_tables"
PAPER_FIGS = ROOT / "figures" / "paper"
PAPER_TABLES.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def _set_style():
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


def load_backtest() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        print(f"ERROR: missing {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    # Drop rows where verification was not possible
    df = df.dropna(subset=["obs_I_total"]).copy()
    df["bias_q50"] = df["I_total_q50"] - df["obs_I_total"]
    df["abs_err_q50"] = df["bias_q50"].abs()
    df["interval_width_90"] = df["I_total_q95"] - df["I_total_q05"]
    return df


# ---------------------------------------------------------------------------
# Agronomic skill metrics
# ---------------------------------------------------------------------------

def kling_gupta_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    """KGE (Kling & Gupta 2009) — 1 is perfect.

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    where r is correlation, alpha = sigma_sim/sigma_obs, beta = mu_sim/mu_obs.
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 3:
        return float("nan")
    obs, sim = obs[valid], sim[valid]
    if obs.std() < 1e-12 or sim.std() < 1e-12 or abs(obs.mean()) < 1e-12:
        return float("nan")
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def percent_bias(obs: np.ndarray, sim: np.ndarray) -> float:
    """PBIAS = 100 * sum(sim - obs) / sum(obs).

    Negative PBIAS = under-prediction. Moriasi et al. (2007) suggest PBIAS in
    [-25%, +25%] is "satisfactory" for streamflow; tighter bands are common
    for soil-water work.
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 3:
        return float("nan")
    obs, sim = obs[valid], sim[valid]
    s = obs.sum()
    if abs(s) < 1e-12:
        return float("nan")
    return float(100 * (sim - obs).sum() / s)


def stress_event_metrics(df: pd.DataFrame, awc_threshold_frac: float = 0.45):
    """Return recall/precision/F1 of irrigation-event prediction.

    A "stress event" here is a 5-day window where the **observed** cumulative
    I exceeds zero (i.e. the deterministic FAO-56 baseline triggered at least
    one irrigation in the horizon). We compare against the model's forecast:
    a "predicted event" is any window where ``I_total_q50 > 0`` *or* where
    ``p_irrigate_tomorrow > 0.5`` (whichever the user prefers — we use both).
    """
    obs_event = df["obs_I_total"] > 0
    pred_event_q50 = df["I_total_q50"] > 0
    # Per the existing summary CSV columns
    if "p_irrigate_tomorrow" in df.columns:
        pred_event_p = df["p_irrigate_tomorrow"] > 0.5
    else:
        pred_event_p = pred_event_q50
    out = {}
    for label, pred in [("q50", pred_event_q50), ("p>0.5", pred_event_p)]:
        tp = int((obs_event & pred).sum())
        fp = int((~obs_event & pred).sum())
        fn = int((obs_event & ~pred).sum())
        tn = int((~obs_event & ~pred).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else float("nan"))
        out[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "recall": recall, "precision": precision, "f1": f1}
    return out


def water_saving_vs_calendar(df: pd.DataFrame, calendar_mm_per_5d: float = 25.0):
    """Compare model recommendation vs a fixed 'calendar' policy.

    A naive calendar policy applies ``calendar_mm_per_5d`` mm every 5 days
    regardless of soil state. The question is: across the backtest, how much
    *less* water does the probabilistic forecast recommend (q50) compared
    to this calendar baseline, while still tracking the FAO-56 deterministic
    requirement?
    """
    n = len(df)
    total_calendar = n * calendar_mm_per_5d
    total_forecast = float(df["I_total_q50"].sum())
    total_observed_demand = float(df["obs_I_total"].sum())
    return {
        "n_5day_windows": n,
        "calendar_mm_total": total_calendar,
        "forecast_q50_mm_total": total_forecast,
        "observed_demand_mm_total": total_observed_demand,
        "saving_vs_calendar_mm": total_calendar - total_forecast,
        "saving_vs_calendar_pct": (
            100 * (total_calendar - total_forecast) / total_calendar),
        "deficit_vs_demand_mm": total_observed_demand - total_forecast,
        "deficit_vs_demand_pct": (
            100 * (total_observed_demand - total_forecast)
            / max(total_observed_demand, 1e-9)),
    }


# ---------------------------------------------------------------------------
# Figure 1 — coverage by city
# ---------------------------------------------------------------------------

def fig_coverage_by_city(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    cov = df.groupby("city")["in_90ci"].mean().sort_values()
    bias = df.groupby("city")["bias_q50"].mean().reindex(cov.index)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    bars = ax.barh(cov.index, cov.values, color="#2c7bb6", alpha=0.8,
                   edgecolor="white")
    ax.axvline(0.90, color="red", ls="--", lw=1.2, label="Nominal 90%")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Coverage of 90% prediction interval")
    ax.set_title("Coverage by city (rolling 5-day backtest)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, cov.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8)

    ax = axes[1]
    colors = ["#a6611a" if b < 0 else "#2c7bb6" for b in bias.values]
    ax.barh(bias.index, bias.values, color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Median forecast bias  (q50 − obs, mm)")
    ax.set_title("Bias by city")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = PAPER_FIGS / "backtest_rolling_coverage_by_city.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — CRPS distribution + per-city box plot
# ---------------------------------------------------------------------------

def fig_crps_distribution(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    crps = df["crps_I_total"].dropna().to_numpy()
    ax.hist(crps, bins=50, color="#2c7bb6", alpha=0.8, edgecolor="white")
    ax.axvline(np.median(crps), color="red", lw=1.5,
               label=f"Median = {np.median(crps):.2f} mm")
    ax.axvline(np.mean(crps), color="orange", ls=":", lw=1.5,
               label=f"Mean = {np.mean(crps):.2f} mm")
    ax.set_xlabel("CRPS — cumulative 5-day irrigation (mm)")
    ax.set_ylabel(f"Frequency  (N = {len(crps)})")
    ax.set_title("CRPS distribution (rolling 5-day backtest)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, np.percentile(crps, 99) * 1.05)

    ax = axes[1]
    cities_sorted = (df.groupby("city")["crps_I_total"].median()
                       .sort_values().index.tolist())
    data_per_city = [df[df["city"] == c]["crps_I_total"].dropna().to_numpy()
                     for c in cities_sorted]
    bp = ax.boxplot(data_per_city, vert=False, labels=cities_sorted,
                    showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red",
                                                "markeredgecolor": "red", "markersize": 5},
                    patch_artist=True, widths=0.6)
    for box in bp["boxes"]:
        box.set_facecolor("#2c7bb6")
        box.set_alpha(0.6)
    ax.set_xlabel("CRPS (mm)")
    ax.set_title("CRPS by city  (◇ = mean,  | = median)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, np.percentile(df["crps_I_total"].dropna(), 99) * 1.05)

    fig.tight_layout()
    out = PAPER_FIGS / "backtest_rolling_crps_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — calibration scatter
# ---------------------------------------------------------------------------

def fig_calibration_scatter(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["obs_I_total"], df["I_total_q50"],
               s=8, alpha=0.25, color="#2c7bb6", label=f"N = {len(df)}")
    lim = max(df["obs_I_total"].max(), df["I_total_q50"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="1:1")
    z = np.polyfit(df["obs_I_total"], df["I_total_q50"], 1)
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, z[0] * xs + z[1], color="red", lw=1.5,
            label=f"Linear fit: y = {z[0]:.2f} x + {z[1]:.1f}")
    ax.set_xlabel("Observed cumulative I  (mm, Xavier baseline)")
    ax.set_ylabel("Forecast median q50  (mm)")
    ax.set_title("Forecast vs observed cumulative 5-day irrigation")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)
    fig.tight_layout()
    out = PAPER_FIGS / "backtest_rolling_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4 — skill as a function of day-of-cycle
# ---------------------------------------------------------------------------

def fig_skill_by_day(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    by_day = df.groupby("day_of_cycle").agg(
        crps_mean=("crps_I_total", "mean"),
        crps_med=("crps_I_total", "median"),
        coverage=("in_90ci", "mean"),
        n=("crps_I_total", "size"),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.plot(by_day["day_of_cycle"], by_day["crps_mean"], color="C0",
            label="CRPS mean")
    ax.plot(by_day["day_of_cycle"], by_day["crps_med"], color="C0", ls="--",
            label="CRPS median")
    ax.set_ylabel("CRPS (mm)")
    ax.set_title("Forecast skill as a function of day-of-cycle  (5-day rolling, all cities, all years)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(by_day["day_of_cycle"], by_day["coverage"], color="C2",
            label="Coverage 90% CI")
    ax.axhline(0.90, color="red", ls="--", lw=1.2, label="Nominal 90%")
    ax.set_xlabel("Day of crop cycle")
    ax.set_ylabel("Coverage 90% CI")
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)

    # Phenological stage bands (5-stage soybean)
    bands = [(0, 15, "Init"), (15, 30, "Dev"), (30, 70, "Mid"),
             (70, 85, "Late"), (85, 90, "Harv")]
    colors_band = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    for (s, e, lbl), c in zip(bands, colors_band):
        for ax in axes:
            ax.axvspan(s, e, alpha=0.08, color=c)
        axes[0].text((s + e) / 2, axes[0].get_ylim()[1] * 0.95, lbl,
                     ha="center", fontsize=8, color="dimgray")

    fig.tight_layout()
    out = PAPER_FIGS / "backtest_rolling_skill_by_day.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Table — per-city summary
# ---------------------------------------------------------------------------

def write_summary_table(df: pd.DataFrame):
    rows = []
    for city in sorted(df["city"].unique()):
        sub = df[df["city"] == city]
        kge = kling_gupta_efficiency(sub["obs_I_total"], sub["I_total_q50"])
        pbias = percent_bias(sub["obs_I_total"], sub["I_total_q50"])
        rows.append({
            "city":         city.replace("_", " "),
            "n_forecasts":  len(sub),
            "crps_mean":    sub["crps_I_total"].mean(),
            "crps_median":  sub["crps_I_total"].median(),
            "mae_q50":      sub["abs_err_q50"].mean(),
            "bias_q50":     sub["bias_q50"].mean(),
            "kge_q50":      kge,
            "pbias_q50":    pbias,
            "coverage_90":  sub["in_90ci"].mean(),
            "ci_width_med": sub["interval_width_90"].median(),
        })
    summary = pd.DataFrame(rows)

    csv_path = PAPER_TABLES / "table_backtest_rolling_summary.csv"
    summary.to_csv(csv_path, index=False, float_format="%.3f")

    agg_kge = kling_gupta_efficiency(df["obs_I_total"], df["I_total_q50"])
    agg_pbias = percent_bias(df["obs_I_total"], df["I_total_q50"])

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-city verification of the rolling 5-day operational "
        r"irrigation forecast \emph{against the FAO-56 deterministic baseline "
        r"driven by Xavier reanalysis observed P and ETo} (not against in-situ "
        r"soil-moisture sensors --- see the validation note in "
        r"\citet{methodology_diagram}). All 5-day forecasts span 2020--2024 "
        r"soybean cycles, day 0 through day 84 (\(N = 425\) per city, "
        r"\(N = 4{,}250\) in aggregate). Skill scores: CRPS is the continuous "
        r"ranked probability score on the cumulative 5-day irrigation (mm); "
        r"MAE, bias, KGE \citep{Kling2009}, and PBIAS \citep{Moriasi2007} are "
        r"on the median forecast \(q_{50}\); coverage is the fraction of "
        r"forecasts whose observed value falls inside the 90\% prediction "
        r"interval. KGE close to 1 indicates good skill; PBIAS in [-25\%, +25\%] "
        r"is considered satisfactory for hydrologic forecasts.}",
        r"\label{tab:backtest-rolling}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"City & N & CRPS$_{mean}$ & MAE & Bias & KGE & PBIAS & Cov$_{90}$ & "
        r"$\Delta_{90}$ \\",
        r" & & (mm) & (mm) & (mm) & & (\%) & & (mm) \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        tex.append(
            f"{r['city']} & {int(r['n_forecasts'])} & "
            f"{r['crps_mean']:.2f} & "
            f"{r['mae_q50']:.2f} & {r['bias_q50']:+.2f} & "
            f"{r['kge_q50']:+.2f} & {r['pbias_q50']:+.1f} & "
            f"{r['coverage_90']:.2%} & {r['ci_width_med']:.1f} \\\\".replace("%", r"\%")
        )
    tex.extend([
        r"\midrule",
        f"\\textbf{{Aggregate}} & {len(df)} & "
        f"{df['crps_I_total'].mean():.2f} & "
        f"{df['abs_err_q50'].mean():.2f} & "
        f"{df['bias_q50'].mean():+.2f} & "
        f"{agg_kge:+.2f} & {agg_pbias:+.1f} & "
        f"{df['in_90ci'].mean():.2%} & "
        f"{df['interval_width_90'].median():.1f} \\\\".replace("%", r"\%"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_path = PAPER_TABLES / "table_backtest_rolling_summary.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")
    return csv_path, tex_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _set_style()
    df = load_backtest()
    agg_kge = kling_gupta_efficiency(df["obs_I_total"], df["I_total_q50"])
    agg_pbias = percent_bias(df["obs_I_total"], df["I_total_q50"])
    stress = stress_event_metrics(df)
    saving = water_saving_vs_calendar(df, calendar_mm_per_5d=25.0)

    print(f"Loaded {len(df)} verifiable forecasts from "
          f"{df['city'].nunique()} cities.")
    print()
    print("=== PROBABILISTIC SKILL ===")
    print(f"  CRPS  mean   : {df['crps_I_total'].mean():.2f} mm")
    print(f"  CRPS  median : {df['crps_I_total'].median():.2f} mm")
    print(f"  Coverage 90% : {df['in_90ci'].mean():.1%}")
    print()
    print("=== POINT FORECAST SKILL (q50 vs deterministic FAO-56 baseline) ===")
    print(f"  MAE          : {df['abs_err_q50'].mean():.2f} mm")
    print(f"  Bias         : {df['bias_q50'].mean():+.2f} mm")
    print(f"  KGE          : {agg_kge:+.3f}  (1 = perfect)")
    print(f"  PBIAS        : {agg_pbias:+.2f} %  (Moriasi: <±25% satisfactory)")
    print()
    print("=== STRESS EVENT DETECTION (5-day window with I > 0 needed) ===")
    for label, m in stress.items():
        print(f"  Decision rule = {label:7s}  "
              f"recall={m['recall']:.2%}  precision={m['precision']:.2%}  "
              f"F1={m['f1']:.2%}  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    print()
    print("=== WATER SAVING vs FIXED 25 mm/5-day CALENDAR ===")
    print(f"  Total forecast q50 : {saving['forecast_q50_mm_total']:.0f} mm "
          f"({saving['n_5day_windows']} windows)")
    print(f"  Total calendar     : {saving['calendar_mm_total']:.0f} mm")
    print(f"  Total demand obs   : {saving['observed_demand_mm_total']:.0f} mm")
    print(f"  Saving vs calendar : {saving['saving_vs_calendar_mm']:.0f} mm "
          f"({saving['saving_vs_calendar_pct']:.1f}%)")
    print(f"  Deficit vs demand  : {saving['deficit_vs_demand_mm']:.0f} mm "
          f"({saving['deficit_vs_demand_pct']:.1f}%)")
    print()

    p1 = fig_coverage_by_city(df);   print(f"  -> {p1.relative_to(ROOT)}")
    p2 = fig_crps_distribution(df);  print(f"  -> {p2.relative_to(ROOT)}")
    p3 = fig_calibration_scatter(df); print(f"  -> {p3.relative_to(ROOT)}")
    p4 = fig_skill_by_day(df);       print(f"  -> {p4.relative_to(ROOT)}")
    csv, tex = write_summary_table(df)
    print(f"  -> {csv.relative_to(ROOT)}")
    print(f"  -> {tex.relative_to(ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
