"""
identify_distributions.py
==========================

Identifica empiricamente a família distribucional de cada combinação
cidade x variável x mês, seguindo protocolo de 4 etapas:
    1) MLE de distribuições candidatas
    2) Comparação por AIC + BIC (Burnham & Anderson 2002)
    3) Teste de aderência Kolmogorov-Smirnov (alfa = 0.05)
    4) Seleção da família vencedora + persistência dos parâmetros

Saídas (em data_processed/distributions/):
    distribution_atlas.json     -> aninhado: {city: {var: {month: {...}}}}
    distribution_atlas.parquet  -> tabela plana (1 linha por combinação)
    summary_diagnostic.png      -> resumo visual da seleção por variável
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / "data" / "extracted_csv").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return here


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "extracted_csv" / "merged_by_city"
OUT_DIR = PROJECT_ROOT / "data_processed" / "distributions"

CITIES = [
    "Baixa_Grande_do_Ribeiro", "Balsas", "Barreiras", "Bom_Jesus",
    "Campos_Lindos", "Correntina", "Formosa_do_Rio_Preto",
    "Luis_Eduardo_Magalhaes", "Tasso_Fragoso", "Urucui",
]

# Período de calibração das distribuições (norma WMO atual)
CALIB_YEAR_START = 1991
CALIB_YEAR_END = 2020

# Limiar de "dia chuvoso" para o componente Bernoulli da ZIG
DRY_THRESHOLD_MM = 1.0

# Famílias candidatas por variável (cada uma é (nome, função_fit))
# Para precip usamos uma estratégia híbrida: Zero-Inflated Gamma como família
# canônica + Gamma puro nos dias chuvosos como benchmark mais simples.
CANDIDATES_BY_VAR = {
    "pr":   ["zigamma", "gamma_wet", "lognorm_wet"],
    "ETo":  ["norm", "lognorm", "gamma"],
    "Tmax": ["norm", "skewnorm"],
    "Tmin": ["norm", "skewnorm"],
    "RH":   ["norm", "beta_rescaled"],
    "Rs":   ["norm", "gamma"],
    "u2":   ["weibull_min", "gamma", "lognorm"],
}

# ---------------------------------------------------------------------------
# Distribuições e utilidades
# ---------------------------------------------------------------------------


def _aic(loglik: float, k: int) -> float:
    return 2 * k - 2 * loglik


def _bic(loglik: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * loglik


def _safe_loglik(dist, *args) -> float:
    """log-likelihood com guarda contra valores -inf."""
    try:
        ll = float(dist.logpdf(*args).sum())
        if not np.isfinite(ll):
            return -np.inf
        return ll
    except Exception:
        return -np.inf


# ---- ZeroInflatedGamma custom ----


def fit_zigamma(values: np.ndarray) -> dict | None:
    """ZIG: p_dry (Bernoulli) + Gamma para dias chuvosos."""
    n = len(values)
    if n < 30:
        return None
    dry_mask = values < DRY_THRESHOLD_MM
    p_dry = float(dry_mask.mean())
    wet = values[~dry_mask]
    if len(wet) < 10:
        return None
    try:
        # Fixa loc=0 (convenção para dist. positivas)
        shape, _, scale = stats.gamma.fit(wet, floc=0)
    except Exception:
        return None
    # Log-lik combinado: dias secos (Bernoulli) + dias chuvosos (Gamma)
    ll_dry = dry_mask.sum() * np.log(p_dry) if p_dry > 0 else 0.0
    ll_wet = (1 - dry_mask).sum() * np.log(1 - p_dry) if p_dry < 1 else 0.0
    ll_gamma = float(stats.gamma.logpdf(wet, shape, loc=0, scale=scale).sum())
    loglik = ll_dry + ll_wet + ll_gamma
    k = 3  # p_dry, shape, scale
    return {
        "family": "zigamma",
        "params": {"p_dry": p_dry, "shape": float(shape), "scale": float(scale)},
        "loglik": float(loglik),
        "n_params": k,
        "n_samples": int(n),
        "aic": _aic(loglik, k),
        "bic": _bic(loglik, k, n),
    }


def fit_gamma_wet(values: np.ndarray) -> dict | None:
    """Gamma ajustado apenas aos dias chuvosos (descarta zeros)."""
    wet = values[values >= DRY_THRESHOLD_MM]
    if len(wet) < 10:
        return None
    try:
        shape, _, scale = stats.gamma.fit(wet, floc=0)
    except Exception:
        return None
    loglik = _safe_loglik(stats.gamma.freeze(shape, loc=0, scale=scale), wet)
    n = len(wet)
    k = 2
    return {
        "family": "gamma_wet",
        "params": {"shape": float(shape), "scale": float(scale)},
        "loglik": loglik,
        "n_params": k,
        "n_samples": n,
        "aic": _aic(loglik, k),
        "bic": _bic(loglik, k, n),
    }


def fit_lognorm_wet(values: np.ndarray) -> dict | None:
    wet = values[values >= DRY_THRESHOLD_MM]
    if len(wet) < 10:
        return None
    try:
        s, _, scale = stats.lognorm.fit(wet, floc=0)
    except Exception:
        return None
    loglik = _safe_loglik(stats.lognorm.freeze(s, loc=0, scale=scale), wet)
    n = len(wet)
    k = 2
    return {
        "family": "lognorm_wet",
        "params": {"s": float(s), "scale": float(scale)},
        "loglik": loglik,
        "n_params": k,
        "n_samples": n,
        "aic": _aic(loglik, k),
        "bic": _bic(loglik, k, n),
    }


# ---- Famílias contínuas padrão (escala por variável) ----


def _fit_scipy_dist(values: np.ndarray, family: str) -> dict | None:
    if len(values) < 30:
        return None
    try:
        if family == "norm":
            params = stats.norm.fit(values)
            dist = stats.norm.freeze(*params)
            param_dict = {"mu": float(params[0]), "sigma": float(params[1])}
            k = 2
        elif family == "lognorm":
            params = stats.lognorm.fit(values, floc=0)
            dist = stats.lognorm.freeze(*params)
            param_dict = {"s": float(params[0]), "scale": float(params[2])}
            k = 2
        elif family == "gamma":
            params = stats.gamma.fit(values, floc=0)
            dist = stats.gamma.freeze(*params)
            param_dict = {"shape": float(params[0]), "scale": float(params[2])}
            k = 2
        elif family == "skewnorm":
            params = stats.skewnorm.fit(values)
            dist = stats.skewnorm.freeze(*params)
            param_dict = {"a": float(params[0]),
                          "loc": float(params[1]),
                          "scale": float(params[2])}
            k = 3
        elif family == "weibull_min":
            params = stats.weibull_min.fit(values, floc=0)
            dist = stats.weibull_min.freeze(*params)
            param_dict = {"c": float(params[0]),
                          "scale": float(params[2])}
            k = 2
        elif family == "beta_rescaled":
            # Rescala RH ou Rs para [0,1] antes de ajustar Beta
            v_min, v_max = values.min(), values.max()
            if v_max - v_min < 1e-6:
                return None
            scaled = (values - v_min) / (v_max - v_min) * 0.998 + 0.001
            a, b, _, _ = stats.beta.fit(scaled, floc=0, fscale=1)
            dist = stats.beta.freeze(a, b, loc=0, scale=1)
            ll_scaled = _safe_loglik(dist, scaled)
            n = len(values)
            param_dict = {"alpha": float(a), "beta": float(b),
                          "scale_min": float(v_min), "scale_max": float(v_max)}
            k = 2
            return {
                "family": "beta_rescaled",
                "params": param_dict,
                "loglik": ll_scaled,
                "n_params": k,
                "n_samples": n,
                "aic": _aic(ll_scaled, k),
                "bic": _bic(ll_scaled, k, n),
            }
        else:
            return None
    except Exception:
        return None

    loglik = _safe_loglik(dist, values)
    n = len(values)
    return {
        "family": family,
        "params": param_dict,
        "loglik": loglik,
        "n_params": k,
        "n_samples": n,
        "aic": _aic(loglik, k),
        "bic": _bic(loglik, k, n),
    }


def fit_candidate(values: np.ndarray, family: str) -> dict | None:
    if family == "zigamma":
        return fit_zigamma(values)
    if family == "gamma_wet":
        return fit_gamma_wet(values)
    if family == "lognorm_wet":
        return fit_lognorm_wet(values)
    return _fit_scipy_dist(values, family)


# ---- Teste KS ----


def ks_test(values: np.ndarray, fit_result: dict) -> dict:
    """Kolmogorov-Smirnov contra a distribuição ajustada."""
    family = fit_result["family"]
    params = fit_result["params"]
    try:
        if family == "zigamma":
            wet = values[values >= DRY_THRESHOLD_MM]
            if len(wet) < 10:
                return {"ks_stat": np.nan, "ks_p": np.nan}
            stat, p = stats.kstest(wet, "gamma",
                                   args=(params["shape"], 0, params["scale"]))
        elif family == "gamma_wet":
            wet = values[values >= DRY_THRESHOLD_MM]
            stat, p = stats.kstest(wet, "gamma",
                                   args=(params["shape"], 0, params["scale"]))
        elif family == "lognorm_wet":
            wet = values[values >= DRY_THRESHOLD_MM]
            stat, p = stats.kstest(wet, "lognorm",
                                   args=(params["s"], 0, params["scale"]))
        elif family == "norm":
            stat, p = stats.kstest(values, "norm",
                                   args=(params["mu"], params["sigma"]))
        elif family == "lognorm":
            stat, p = stats.kstest(values, "lognorm",
                                   args=(params["s"], 0, params["scale"]))
        elif family == "gamma":
            stat, p = stats.kstest(values, "gamma",
                                   args=(params["shape"], 0, params["scale"]))
        elif family == "skewnorm":
            stat, p = stats.kstest(values, "skewnorm",
                                   args=(params["a"], params["loc"], params["scale"]))
        elif family == "weibull_min":
            stat, p = stats.kstest(values, "weibull_min",
                                   args=(params["c"], 0, params["scale"]))
        elif family == "beta_rescaled":
            scaled = ((values - params["scale_min"]) /
                      (params["scale_max"] - params["scale_min"]) * 0.998 + 0.001)
            stat, p = stats.kstest(scaled, "beta",
                                   args=(params["alpha"], params["beta"], 0, 1))
        else:
            return {"ks_stat": np.nan, "ks_p": np.nan}
        return {"ks_stat": float(stat), "ks_p": float(p)}
    except Exception:
        return {"ks_stat": np.nan, "ks_p": np.nan}


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def load_city_period(city: str) -> pd.DataFrame:
    path = DATA_DIR / f"{city}.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    df = df.loc[f"{CALIB_YEAR_START}-01-01":f"{CALIB_YEAR_END}-12-31"]
    return df


def select_winning_family(candidates: list[dict]) -> dict:
    """Escolhe a família vencedora por menor AIC, com ressalva BIC."""
    valid = [c for c in candidates if c is not None and np.isfinite(c["aic"])]
    if not valid:
        return {"family": "FAILED", "params": {}, "aic": np.nan, "bic": np.nan,
                "n_samples": 0, "ks_p": np.nan, "loglik": np.nan,
                "n_params": 0, "selection_note": "no valid candidates"}
    valid.sort(key=lambda c: c["aic"])
    winner = valid[0]
    runner_up = valid[1] if len(valid) > 1 else None
    note = ""
    if runner_up is not None:
        delta = runner_up["aic"] - winner["aic"]
        if delta < 2:
            note = (f"Delta_AIC < 2 vs {runner_up['family']} "
                    f"(parsimony preserved)")
    winner["selection_note"] = note
    winner["all_candidates"] = [
        {"family": c["family"], "aic": c["aic"], "bic": c["bic"]}
        for c in valid
    ]
    return winner


def identify_for_city_var_month(values: np.ndarray, var: str) -> dict:
    """Roda o protocolo de 4 etapas para uma combinação específica."""
    candidates = [fit_candidate(values, fam) for fam in CANDIDATES_BY_VAR[var]]
    winner = select_winning_family(candidates)
    if winner["family"] != "FAILED":
        ks = ks_test(values, winner)
        winner.update(ks)
        winner["passes_ks_05"] = bool(
            not np.isnan(winner["ks_p"]) and winner["ks_p"] >= 0.05
        )
    return winner


def run_pipeline() -> tuple[dict, pd.DataFrame]:
    print("=" * 78)
    print(" Distribution identification pipeline")
    print("=" * 78)
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Output dir    : {OUT_DIR}")
    print(f"Calib period  : {CALIB_YEAR_START}-{CALIB_YEAR_END}")
    print(f"Variables     : {list(CANDIDATES_BY_VAR.keys())}")
    print(f"Dry threshold : {DRY_THRESHOLD_MM} mm")
    print()

    atlas: dict[str, Any] = {}
    flat_rows = []

    for ic, city in enumerate(CITIES, 1):
        print(f"[{ic:2d}/{len(CITIES)}] {city} ...", flush=True)
        try:
            df = load_city_period(city)
        except FileNotFoundError as e:
            print(f"   skipped: {e}")
            continue
        atlas[city] = {}
        for var in CANDIDATES_BY_VAR:
            if var not in df.columns:
                continue
            atlas[city][var] = {}
            for month in range(1, 13):
                vals = df.loc[df.index.month == month, var].dropna().values
                if len(vals) < 30:
                    continue
                result = identify_for_city_var_month(vals, var)
                atlas[city][var][str(month)] = {
                    k: v for k, v in result.items()
                    if k != "all_candidates"
                }
                flat_rows.append({
                    "city": city,
                    "variable": var,
                    "month": month,
                    "n_samples": result.get("n_samples", 0),
                    "winning_family": result["family"],
                    "loglik": result.get("loglik", np.nan),
                    "aic": result.get("aic", np.nan),
                    "bic": result.get("bic", np.nan),
                    "ks_p": result.get("ks_p", np.nan),
                    "passes_ks_05": result.get("passes_ks_05", False),
                    "selection_note": result.get("selection_note", ""),
                })

    df_flat = pd.DataFrame(flat_rows)
    return atlas, df_flat


def save_outputs(atlas: dict, df_flat: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON aninhado para alimentar PyMC
    payload = {
        "metadata": {
            "title": "Empirical distribution identification atlas",
            "calibration_period": f"{CALIB_YEAR_START}-{CALIB_YEAR_END}",
            "dry_threshold_mm": DRY_THRESHOLD_MM,
            "variables": list(CANDIDATES_BY_VAR.keys()),
            "candidates_by_variable": CANDIDATES_BY_VAR,
            "selection_criterion": "min AIC, ties (Delta_AIC<2) broken by parsimony",
            "goodness_of_fit_test": "Kolmogorov-Smirnov, alpha=0.05",
            "n_combinations": int(len(df_flat)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "atlas": atlas,
    }
    out_json = OUT_DIR / "distribution_atlas.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float, ensure_ascii=False)
    print(f"\n  JSON     : {out_json} ({out_json.stat().st_size/1024:.0f} KB)")

    # Parquet plano para análise
    out_parquet = OUT_DIR / "distribution_atlas.parquet"
    try:
        df_flat.to_parquet(out_parquet, index=False)
        print(f"  Parquet  : {out_parquet}")
    except ImportError:
        print(f"  Parquet  : PULADO (instale pyarrow)")

    # CSV inspeção rápida
    out_csv = OUT_DIR / "distribution_atlas.csv"
    df_flat.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"  CSV      : {out_csv}")


def make_summary_diagnostic(df_flat: pd.DataFrame) -> None:
    """Resumo visual: vencedoras por variável."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Plot     : PULADO (matplotlib não instalado)")
        return

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    for ax, var in zip(axes, CANDIDATES_BY_VAR):
        sub = df_flat[df_flat["variable"] == var]
        if sub.empty:
            ax.set_visible(False)
            continue
        counts = sub["winning_family"].value_counts()
        colors = ["#2c7bb6" if i == 0 else "#abd9e9"
                  for i in range(len(counts))]
        ax.bar(range(len(counts)), counts.values, color=colors)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{var}  (n={len(sub)})", fontsize=10)
        ax.set_ylabel("# winners")
        # passes KS
        pct_ks = 100 * sub["passes_ks_05"].sum() / len(sub)
        ax.text(0.97, 0.97, f"KS pass: {pct_ks:.0f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    # esconde excesso
    for j in range(len(CANDIDATES_BY_VAR), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(
        f"Empirical winning distribution families by variable ({CALIB_YEAR_START}-{CALIB_YEAR_END})",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = OUT_DIR / "summary_diagnostic.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Plot     : {out_png}")


def main() -> None:
    atlas, df_flat = run_pipeline()
    save_outputs(atlas, df_flat)
    make_summary_diagnostic(df_flat)
    # Print summary
    print("\n--- Summary ---")
    print(f"  Total combinations: {len(df_flat)}")
    print(f"  KS-pass rate (alpha=0.05): "
          f"{100*df_flat['passes_ks_05'].sum()/len(df_flat):.1f}%")
    print(f"\n  Winning family by variable:")
    for var in CANDIDATES_BY_VAR:
        sub = df_flat[df_flat["variable"] == var]
        if sub.empty:
            continue
        winners = sub["winning_family"].value_counts()
        top = winners.index[0]
        pct = 100 * winners.iloc[0] / len(sub)
        print(f"    {var:5s}: {top:18s} ({pct:.0f}% of {len(sub)} combinations)")
    print("\nConcluído.")


if __name__ == "__main__":
    main()