"""Compare Bayesian Dirichlet-Multinomial SPEI forecasts against Xavier.

Generates the validation artefacts the manuscript needs:

* ``output/paper_tables/bayes_vs_xavier.csv`` -- per-cycle predicted class
  probabilities, predicted class (argmax of Dirichlet prior alpha *before*
  observing the cycle), observed class, and forecast vs Xavier irrigation.
* ``output/paper_tables/confusion_matrix.csv`` and ``.tex`` -- 3x3 confusion
  matrix with per-class precision/recall/F1 + global accuracy + Cohen's kappa.
* ``output/paper_tables/roc_metrics.csv`` -- per-class one-vs-rest AUC + the
  operational dry-vs-rest AUC, both with 95% bootstrap CI.
* ``figures/paper/roc_one_vs_rest.png`` -- 3 ROC curves (dry, normal, wet).
* ``figures/paper/roc_dry_vs_rest.png`` -- operational drought-vs-rest ROC.
* ``figures/paper/reliability_diagram.png`` -- multiclass reliability curve.
* ``figures/paper/pit_histogram.png`` -- PIT histogram of forecast irrigation.
* ``output/paper_tables/table1_cities.csv`` and ``.tex`` -- 10-city
  characterisation (lat/lon, Koppen, AWC, P_bar, ETo_bar, aridity, ENSO).

Method note
-----------
The sequential pipeline only persists the *final* Dirichlet alpha per city
(``alpha_final_<city>.json``). The per-cycle prior alpha is reconstructed
analytically: alpha_prior(t) = alpha_0 + sum(observed_class counts before t),
with alpha_0 = (1, 1, 1) (the default in :func:`bwb.cli` and
:func:`bwb.forecast.climatological.run_sequential_forecast`). This matches
exactly the alpha used inside the sampler at cycle t, so the predicted-class
probabilities here are the same the sampler used to generate the forecast.

Usage
-----
::

    python scripts/compare_bayes_vs_xavier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "output" / "forecast_sequential" / "sequential_summary_all_cities.csv"
ALPHA_DIR = ROOT / "output" / "forecast_sequential"
PAPER_TABLES = ROOT / "output" / "paper_tables"
PAPER_FIGS = ROOT / "figures" / "paper"
PAPER_TABLES.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["dry", "normal", "wet"]
N_CLASSES = 3
ALPHA_0 = np.ones(N_CLASSES, dtype=float)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Per-cycle predicted class probabilities
# ---------------------------------------------------------------------------

def _cycle_start_year(cycle_str: str) -> int:
    return int(cycle_str.split("/")[0])


def derive_predicted_probabilities(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (city, cycle) compute the Dirichlet prior alpha used by the
    sampler when forecasting that cycle, and return the expected class
    probabilities + argmax predicted class.

    Verification: the alpha_post after the last cycle reconstructed here must
    equal the value persisted in ``alpha_final_<city>.json``.
    """
    rows = []
    for city, group in summary.groupby("city"):
        group = group.sort_values("cycle").reset_index(drop=True)
        alpha = ALPHA_0.copy()
        for _, row in group.iterrows():
            # alpha here is the prior BEFORE observing this cycle
            probs = alpha / alpha.sum()
            predicted = int(np.argmax(alpha))
            observed = int(row["observed_class"])
            rows.append({
                "city": city,
                "cycle": row["cycle"],
                "cycle_start": _cycle_start_year(row["cycle"]),
                "alpha_prior_dry": float(alpha[0]),
                "alpha_prior_normal": float(alpha[1]),
                "alpha_prior_wet": float(alpha[2]),
                "P_dry": float(probs[0]),
                "P_normal": float(probs[1]),
                "P_wet": float(probs[2]),
                "predicted_class": predicted,
                "observed_class": observed,
                "hit": int(predicted == observed),
            })
            # update alpha with the observed class for the next cycle
            alpha[observed] += 1.0

        # sanity check vs persisted alpha_final
        alpha_final_path = ALPHA_DIR / f"alpha_final_{city}.json"
        if alpha_final_path.exists():
            persisted = np.asarray(json.loads(alpha_final_path.read_text()), dtype=float)
            if not np.allclose(persisted, alpha):
                print(
                    f"  [warn] {city}: reconstructed alpha {alpha.tolist()} "
                    f"!= persisted {persisted.tolist()}"
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Confusion matrix + Cohen's kappa + per-class metrics
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def cohens_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if 1 - pe == 0:
        return float("nan")
    return float((po - pe) / (1 - pe))


def per_class_metrics(cm: np.ndarray) -> pd.DataFrame:
    """Returns precision, recall, F1, support per class."""
    rows = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        support = cm[k, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "class": CLASS_NAMES[k],
            "support": int(support),
            "precision": prec,
            "recall": rec,
            "F1": f1,
        })
    return pd.DataFrame(rows)


def write_confusion_artefacts(table: pd.DataFrame) -> dict:
    y_true = table["observed_class"].to_numpy()
    y_pred = table["predicted_class"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, N_CLASSES)
    accuracy = float(np.trace(cm) / cm.sum())
    kappa = cohens_kappa(cm)
    per_class = per_class_metrics(cm)

    cm_df = pd.DataFrame(
        cm,
        index=[f"obs_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES],
    )
    cm_df.to_csv(PAPER_TABLES / "confusion_matrix.csv")
    per_class.to_csv(PAPER_TABLES / "per_class_metrics.csv", index=False)

    # LaTeX confusion matrix
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Confusion matrix of SPEI class predictions vs Xavier reanalysis "
        r"(\(N=" + str(int(cm.sum())) + r"\) crop cycles, 10 cities $\times$ 5 "
        r"safras 2020-2024). Accuracy = " + f"{accuracy:.2f}" + r", Cohen's "
        r"$\kappa$ = " + f"{kappa:.2f}" + r".}",
        r"\label{tab:confusion-matrix}",
        r"\begin{tabular}{lccc|c}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Predicted} & \\",
        r"Observed & dry & normal & wet & support \\",
        r"\midrule",
    ]
    for k, name in enumerate(CLASS_NAMES):
        row = cm[k]
        tex_lines.append(
            f"{name} & {row[0]} & {row[1]} & {row[2]} & {int(row.sum())} \\\\"
        )
    tex_lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Per-class precision/recall/F1}}\\",
    ])
    for _, r in per_class.iterrows():
        tex_lines.append(
            f"{r['class']} & {r['precision']:.2f} & {r['recall']:.2f} "
            f"& {r['F1']:.2f} & \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    (PAPER_TABLES / "confusion_matrix.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8"
    )
    return {"accuracy": accuracy, "kappa": kappa, "cm": cm}


# ---------------------------------------------------------------------------
# 3. ROC + bootstrap AUC
# ---------------------------------------------------------------------------

def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard ROC curve (binary). Returns (fpr, tpr) sorted by threshold."""
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    return fpr, tpr


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    trap = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renames trapz
    return float(trap(tpr, fpr))


def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        # need both classes in the resample to define ROC
        if yt.sum() == 0 or yt.sum() == n:
            continue
        fpr, tpr = roc_curve(yt, ys)
        aucs.append(auc_trapezoid(fpr, tpr))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(aucs)
    return float(np.median(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def write_roc_artefacts(table: pd.DataFrame) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    y_obs = table["observed_class"].to_numpy()
    P = table[["P_dry", "P_normal", "P_wet"]].to_numpy()

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- one-vs-rest ROC ---
    ax = axes[0]
    colors = {"dry": "#a6611a", "normal": "#7f7f7f", "wet": "#2c7bb6"}
    for k, name in enumerate(CLASS_NAMES):
        y_true = (y_obs == k).astype(int)
        y_score = P[:, k]
        fpr, tpr = roc_curve(y_true, y_score)
        auc_pt = auc_trapezoid(fpr, tpr)
        auc_med, lo, hi = bootstrap_auc(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_pt:.2f}; 95% CI [{lo:.2f}, {hi:.2f}])",
                color=colors[name], lw=1.8)
        rows.append({
            "class": name, "scheme": "one-vs-rest",
            "AUC": auc_pt, "AUC_boot_median": auc_med,
            "AUC_CI95_lo": lo, "AUC_CI95_hi": hi,
            "n_pos": int(y_true.sum()), "n_neg": int(len(y_true) - y_true.sum()),
        })
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC (3 SPEI classes)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # --- dry-vs-rest (operational drought ROC) ---
    ax = axes[1]
    y_true_dry = (y_obs == 0).astype(int)
    y_score_dry = P[:, 0]
    fpr, tpr = roc_curve(y_true_dry, y_score_dry)
    auc_pt = auc_trapezoid(fpr, tpr)
    auc_med, lo, hi = bootstrap_auc(y_true_dry, y_score_dry)
    ax.plot(fpr, tpr, color="#a6611a", lw=2.0,
            label=f"AUC={auc_pt:.2f}; 95% CI [{lo:.2f}, {hi:.2f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Drought (dry) vs rest ROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    rows.append({
        "class": "dry", "scheme": "dry-vs-rest",
        "AUC": auc_pt, "AUC_boot_median": auc_med,
        "AUC_CI95_lo": lo, "AUC_CI95_hi": hi,
        "n_pos": int(y_true_dry.sum()),
        "n_neg": int(len(y_true_dry) - y_true_dry.sum()),
    })

    fig.tight_layout()
    fig.savefig(PAPER_FIGS / "roc_one_vs_rest.png")
    plt.close(fig)

    roc_df = pd.DataFrame(rows)
    roc_df.to_csv(PAPER_TABLES / "roc_metrics.csv", index=False)
    return roc_df


# ---------------------------------------------------------------------------
# 4. Reliability diagram + PIT histogram
# ---------------------------------------------------------------------------

def write_reliability_and_pit(table: pd.DataFrame, summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    # --- Reliability diagram (probability of true class) ---
    P = table[["P_dry", "P_normal", "P_wet"]].to_numpy()
    y_obs = table["observed_class"].to_numpy()
    p_true = P[np.arange(len(table)), y_obs]      # P(class = observed)
    obs_indicator = np.ones_like(p_true)          # always 1: the true class did occur

    # Pool predicted prob across (cycle, class) for full reliability
    # Each (cycle, class) is a (predicted_prob, occurred?) point
    flat_p = np.concatenate([P[:, k] for k in range(N_CLASSES)])
    flat_y = np.concatenate([(y_obs == k).astype(int) for k in range(N_CLASSES)])

    bins = np.linspace(0, 1, 11)        # 10 bins of width 0.1
    bin_idx = np.clip(np.digitize(flat_p, bins) - 1, 0, len(bins) - 2)
    centers, mean_p, freq, counts = [], [], [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        mean_p.append(flat_p[mask].mean())
        freq.append(flat_y[mask].mean())
        counts.append(int(mask.sum()))
    centers = np.asarray(centers)
    mean_p = np.asarray(mean_p)
    freq = np.asarray(freq)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="Perfect calibration")
    ax.plot(mean_p, freq, "o-", color="#2c7bb6", lw=1.8, ms=6,
            label="Observed frequency")
    for c, mp, fr, cnt in zip(centers, mean_p, freq, counts):
        ax.annotate(f"n={cnt}", (mp, fr), xytext=(4, -10),
                    textcoords="offset points", fontsize=7, color="grey")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (pooled SPEI classes)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # --- PIT histogram (irrigation total CRPS-PIT already in summary) ---
    ax = axes[1]
    pit = summary["prob_PIT_I_total"].dropna().to_numpy()
    n = len(pit)
    nbins = 10
    counts_pit, edges = np.histogram(pit, bins=np.linspace(0, 1, nbins + 1))
    expected = n / nbins
    centers_pit = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers_pit, counts_pit, width=1 / nbins * 0.9,
           color="#2c7bb6", alpha=0.8, edgecolor="white")
    ax.axhline(expected, color="red", lw=1.2, ls="--",
               label=f"Uniform expectation (n/{nbins} = {expected:.1f})")
    ax.set_xlabel(r"PIT for total irrigation $I_{\mathrm{total}}$")
    ax.set_ylabel("Count")
    ax.set_title(f"PIT histogram (N = {n} forecasts)")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(PAPER_FIGS / "reliability_pit.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Table 1 -- 10 cities characterisation
# ---------------------------------------------------------------------------

def _load_matopiba_toml() -> dict:
    try:
        import tomllib  # Python 3.11+
    except ImportError:                 # pragma: no cover
        import tomli as tomllib
    path = ROOT / "src" / "bwb" / "config" / "regional" / "matopiba.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def _classify_safra_enso(oni: pd.DataFrame, year_start: int) -> str:
    """Classify a soybean safra (Dec year_start - Mar year_start+1) by ONI.

    Conventional thresholds (NOAA CPC):
        El Nino : ONI >= +0.5 over 5 consecutive overlapping seasons
        La Nina : ONI <= -0.5
    Here we simplify to the *DJF* mean ONI of the safra start year.
    """
    djf = oni[(oni["year"] == year_start) & (oni["month"] == 12) |
              (oni["year"] == year_start + 1) & (oni["month"].isin([1, 2]))]
    if djf.empty:
        return "unknown"
    mean = djf["oni"].mean()
    if mean >= 0.5:
        return "El Nino"
    if mean <= -0.5:
        return "La Nina"
    return "Neutral"


def write_table1(summary: pd.DataFrame):
    profile = _load_matopiba_toml()
    cities_coords: dict[str, list[float]] = profile["cities"]
    awc_mm = profile["crop"]["awc_mm"]

    normals = pd.read_csv(ROOT / "data_processed" / "climatological_normals" / "csv" /
                           "annual_normals.csv")
    normals_recent = normals[normals["period"] == "1991-2020"]
    pr_table = normals_recent[normals_recent["variable"] == "pr"].set_index("city")["mean"]
    eto_table = normals_recent[normals_recent["variable"] == "ETo"].set_index("city")["mean"]

    oni = pd.read_csv(ROOT / "data_processed" / "oceanic_processed" / "oni.csv")

    # ENSO frequency over 1961-2020 DJF
    djf = oni[oni["month"].isin([12, 1, 2])].copy()
    djf["safra"] = np.where(djf["month"] == 12, djf["year"], djf["year"] - 1)
    djf_mean = djf.groupby("safra")["oni"].mean().reset_index()
    djf_mean = djf_mean[(djf_mean["safra"] >= 1961) & (djf_mean["safra"] <= 2020)]
    n_total = len(djf_mean)
    n_nino = int((djf_mean["oni"] >= 0.5).sum())
    n_nina = int((djf_mean["oni"] <= -0.5).sum())
    n_neut = n_total - n_nino - n_nina
    enso_clim = (
        f"El Nino {n_nino}/{n_total} ({100*n_nino/n_total:.0f}\\%), "
        f"La Nina {n_nina}/{n_total} ({100*n_nina/n_total:.0f}\\%), "
        f"Neutral {n_neut}/{n_total} ({100*n_neut/n_total:.0f}\\%)"
    )

    # Per-city ENSO over the 5 study safras (2020-2024)
    safra_years = [2020, 2021, 2022, 2023, 2024]
    safra_enso = {y: _classify_safra_enso(oni, y) for y in safra_years}

    rows = []
    for city, (lat, lon) in cities_coords.items():
        p_bar = float(pr_table.get(city, np.nan))
        eto_bar = float(eto_table.get(city, np.nan))
        aridity = p_bar / eto_bar if eto_bar > 0 else np.nan
        rows.append({
            "city": city,
            "lat": lat,
            "lon": lon,
            "Koppen": "Aw",                    # MATOPIBA-wide tropical savanna
            "AWC_mm": awc_mm,
            "P_annual_1991_2020_mm": round(p_bar, 1),
            "ETo_annual_1991_2020_mm": round(eto_bar, 1),
            "P_over_ETo": round(aridity, 3),
            "n_safras_El_Nino_2020_2024": sum(
                1 for y in safra_years if safra_enso[y] == "El Nino"),
            "n_safras_La_Nina_2020_2024": sum(
                1 for y in safra_years if safra_enso[y] == "La Nina"),
            "n_safras_Neutral_2020_2024": sum(
                1 for y in safra_years if safra_enso[y] == "Neutral"),
        })
    table = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)
    table.to_csv(PAPER_TABLES / "table1_cities.csv", index=False)

    # LaTeX
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{The 10 MATOPIBA municipalities used in the validation grid. "
        r"$\bar{P}$ and $\bar{ETo}$ are 1991-2020 annual means (Xavier "
        r"reanalysis). Aridity = $\bar{P}/\bar{ETo}$. AWC is the FAO-56 "
        r"available water capacity over the 60\,cm root zone, set uniformly "
        r"at " + str(awc_mm) + r"\,mm. Koppen class is Aw (tropical savanna) "
        r"for the entire region. ENSO regime of the five study safras "
        r"(Dec--Mar): " + ", ".join(
            f"{y}/{y+1}: {safra_enso[y]}" for y in safra_years) + r". "
        r"Long-term ENSO frequency 1961--2020 (DJF ONI): " + enso_clim + ".}",
        r"\label{tab:study-cities}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"City & Lat & Lon & $\bar{P}$ (mm) & $\bar{ETo}$ (mm) & $\bar{P}/\bar{ETo}$ \\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        tex.append(
            f"{r['city'].replace('_', ' ')} & "
            f"{r['lat']:.2f} & {r['lon']:.2f} & "
            f"{r['P_annual_1991_2020_mm']:.0f} & "
            f"{r['ETo_annual_1991_2020_mm']:.0f} & "
            f"{r['P_over_ETo']:.2f} \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    (PAPER_TABLES / "table1_cities.tex").write_text("\n".join(tex), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _set_style()
    if not SUMMARY_CSV.exists():
        print(f"ERROR: missing {SUMMARY_CSV}", file=sys.stderr)
        sys.exit(1)

    summary = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(summary)} forecast rows from {SUMMARY_CSV.name}")

    # 1) per-cycle predicted probabilities + classes
    table = derive_predicted_probabilities(summary)
    merged = summary.merge(
        table.drop(columns=["observed_class"]),
        on=["city", "cycle"], how="left",
    )
    cols = [
        "city", "cycle", "observed_class", "predicted_class", "hit",
        "P_dry", "P_normal", "P_wet",
        "alpha_prior_dry", "alpha_prior_normal", "alpha_prior_wet",
        "det_I_total_obs_mm", "prob_I_total_q05", "prob_I_total_q50",
        "prob_I_total_q95", "det_I_total_error_mm",
        "prob_CRPS_I_total_mm", "prob_PIT_I_total",
        "prob_coverage_90_I_total", "prob_coverage_90_SW_daily",
        "det_KGE_SW", "det_NSE_SW", "det_Pearson_r_SW", "det_MAE_SW_mm",
        "CRPSS_vs_naive_climatology", "CRPSS_vs_persistence",
        "CRPSS_vs_long_term_mean",
    ]
    cols = [c for c in cols if c in merged.columns]
    out = merged[cols]
    out_path = PAPER_TABLES / "bayes_vs_xavier.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> {out_path.relative_to(ROOT)} ({len(out)} rows)")

    # 2) confusion matrix + kappa
    cm_info = write_confusion_artefacts(table)
    print(f"  -> confusion_matrix.csv/.tex  acc={cm_info['accuracy']:.3f}  "
          f"kappa={cm_info['kappa']:.3f}")

    # 3) ROC + bootstrap
    roc_df = write_roc_artefacts(table)
    print("  -> roc_metrics.csv + figures/paper/roc_one_vs_rest.png")
    for _, r in roc_df.iterrows():
        print(f"     {r['scheme']:14s} {r['class']:8s} "
              f"AUC={r['AUC']:.2f}  CI=[{r['AUC_CI95_lo']:.2f}, "
              f"{r['AUC_CI95_hi']:.2f}]")

    # 4) reliability + PIT
    write_reliability_and_pit(table, summary)
    print("  -> figures/paper/reliability_pit.png")

    # 5) Table 1
    write_table1(summary)
    print("  -> table1_cities.csv/.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()
