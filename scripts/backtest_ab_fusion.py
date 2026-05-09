"""A/B backtest: Xavier-only vs NASA+OpenMeteo-Archive fused.

Goal
----
Quantify the skill change of the rolling 5-day forecast when we replace
the curated Xavier reanalysis (the only source available for the
1961-2019 backtest history) with the NASA POWER + Open-Meteo Archive
fused product that will drive operations in 2026.

Design choice — equal-history A/B
---------------------------------
Xavier covers 1961-present, but NASA POWER and Archive only start in
1990. To isolate the *source* effect from the *amount-of-history*
effect, both arms are limited to 1990-onwards:

* Arm A — Xavier only, 1990-01-01 .. forecast_date.
* Arm B — Fused (NASA POWER + Archive), 1990-01-01 .. forecast_date,
  with ETo recomputed via FAO-56 Penman-Monteith on the fused
  Tmax/Tmin/RH/u2/Rs (see :func:`bwb.data.sources.climate.fuse_climate_sources`).

Scope
-----
Single city (Balsas/MA), 5 cycles × 85 day_of_cycle = 425 forecasts
per arm. NASA POWER + Archive are downloaded once (35-year window) and
cached locally as parquet. Total wall time on a warm cache: ~90 s.

Outputs
-------
* ``output/backtest_ab_fusion/balsas_ab.csv`` — per-forecast metrics
  for both arms, side-by-side.
* ``output/backtest_ab_fusion/balsas_ab_summary.csv`` — aggregate
  comparison table (CRPS, MAE, KGE, PBIAS, coverage).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bwb.data.loaders import load_city_series
from bwb.data.sources.climate import (
    download_nasa_power,
    download_openmeteo_archive,
    fuse_climate_sources,
)
from bwb.forecast.rolling import rolling_5day_forecast
from bwb.phenology.kc_curves import soybean_kc_90d_step

CITY = "Balsas"
LAT, LON, ELEV = -7.53, -46.04, 285.0
HISTORY_START = date(1990, 1, 1)
HISTORY_END = date(2025, 3, 1)
AWC_MM = 120.0
N_SIM = 200          # smaller than the 500 used in the main backtest;
                      # the A/B comparison is paired so noise cancels.
RANDOM_SEED = 42

OUT_DIR = ROOT / "output" / "backtest_ab_fusion"
CACHE_DIR = ROOT / "output" / "backtest_ab_fusion" / "_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (with on-disk cache)
# ---------------------------------------------------------------------------


def load_xavier_1990plus() -> pd.DataFrame:
    df = load_city_series(CITY)
    df["date"] = pd.to_datetime(df["date"])
    return df.loc[df["date"] >= pd.Timestamp(HISTORY_START),
                  ["date", "pr", "ETo"]].reset_index(drop=True)


def _cached_download(name: str, builder) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    df = builder()
    df.to_parquet(path)
    return df


def load_fused_1990plus() -> pd.DataFrame:
    nasa = _cached_download(
        f"nasa_{CITY}",
        lambda: download_nasa_power(
            lat=LAT, lon=LON,
            start=HISTORY_START, end=HISTORY_END,
            elevation_m=ELEV,
        ),
    )
    arch = _cached_download(
        f"archive_{CITY}",
        lambda: download_openmeteo_archive(
            lat=LAT, lon=LON,
            start=HISTORY_START, end=HISTORY_END,
        ),
    )
    fused = fuse_climate_sources(
        {"nasa_power": nasa, "openmeteo_archive": arch},
        lat=LAT, elevation_m=ELEV,
    )
    return fused[["date", "pr", "ETo"]].copy()


# ---------------------------------------------------------------------------
# Per-cycle backtest
# ---------------------------------------------------------------------------


def run_cycle(history_df: pd.DataFrame, planting_date: pd.Timestamp,
              kc: np.ndarray, *, label: str) -> pd.DataFrame:
    rows = []
    for d in range(85):
        forecast_date = planting_date + pd.Timedelta(days=d)
        try:
            rf = rolling_5day_forecast(
                history_df=history_df,
                planting_date=planting_date,
                forecast_date=forecast_date,
                kc_curve=kc,
                awc_mm=AWC_MM,
                cycle_days=90,
                n_simulations=N_SIM,
                random_seed=RANDOM_SEED,
            )
        except Exception as exc:
            rows.append({"forecast_date": forecast_date,
                         "day_of_cycle": d,
                         "arm": label,
                         "error": str(exc)})
            continue
        rows.append({
            "forecast_date": forecast_date,
            "day_of_cycle": d,
            "arm": label,
            "SW_today_mm": rf.SW_today_mm,
            "I_total_q05": rf.I_total_horizon_q05,
            "I_total_q50": rf.I_total_horizon_q50,
            "I_total_q95": rf.I_total_horizon_q95,
            "p_irrigate_tomorrow": rf.p_irrigate_tomorrow,
            "obs_I_total": rf.obs_I_total_horizon,
            "crps_I_total": rf.crps_I_total,
            "in_90ci": (
                rf.I_total_horizon_q05 <= rf.obs_I_total_horizon <= rf.I_total_horizon_q95
                if rf.obs_I_total_horizon is not None else None),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Skill metrics (mirror of analyze_backtest_rolling.py)
# ---------------------------------------------------------------------------


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 3:
        return float("nan")
    obs, sim = obs[m], sim[m]
    if obs.std() < 1e-12 or sim.std() < 1e-12 or abs(obs.mean()) < 1e-12:
        return float("nan")
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 3 or abs(obs[m].sum()) < 1e-12:
        return float("nan")
    return float(100 * (sim[m] - obs[m]).sum() / obs[m].sum())


def summarise(df: pd.DataFrame, *, arm: str) -> dict:
    sub = df[df["arm"] == arm].dropna(subset=["obs_I_total"])
    err = (sub["I_total_q50"] - sub["obs_I_total"]).to_numpy()
    return {
        "arm": arm,
        "n": len(sub),
        "crps_mean": float(sub["crps_I_total"].mean()),
        "crps_median": float(sub["crps_I_total"].median()),
        "mae_q50": float(np.mean(np.abs(err))),
        "bias_q50": float(np.mean(err)),
        "kge_q50": kge(sub["obs_I_total"], sub["I_total_q50"]),
        "pbias_q50": pbias(sub["obs_I_total"], sub["I_total_q50"]),
        "coverage_90": float(sub["in_90ci"].astype(float).mean()),
        "ci_width_med": float((sub["I_total_q95"] - sub["I_total_q05"]).median()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"[1/4] loading Xavier 1990+ for {CITY}")
    xavier = load_xavier_1990plus()
    print(f"      {len(xavier)} daily rows ({xavier['date'].min().date()}..{xavier['date'].max().date()})")

    print(f"[2/4] loading NASA + Archive (fused) 1990+ for {CITY}")
    fused = load_fused_1990plus()
    print(f"      {len(fused)} daily rows ({fused['date'].min().date()}..{fused['date'].max().date()})")

    kc = soybean_kc_90d_step()

    pieces = []
    for cy in [2020, 2021, 2022, 2023, 2024]:
        plant = pd.Timestamp(year=cy, month=12, day=1)
        print(f"[3/4] cycle {cy}: arm A (Xavier)…")
        pieces.append(run_cycle(xavier, plant, kc, label="xavier_1990plus")
                      .assign(cycle_year=cy))
        print(f"[3/4] cycle {cy}: arm B (Fused)…")
        pieces.append(run_cycle(fused, plant, kc, label="fused_nasa_archive")
                      .assign(cycle_year=cy))
    full = pd.concat(pieces, ignore_index=True)
    full.to_csv(OUT_DIR / "balsas_ab.csv", index=False)
    print(f"[4/4] wrote {OUT_DIR/'balsas_ab.csv'} ({len(full)} rows)")

    print()
    print("=" * 96)
    print(f"A/B summary — {CITY}, 5 cycles × 85 day_of_cycle = 425 forecasts per arm")
    print("=" * 96)
    rows = [summarise(full, arm=a) for a in ["xavier_1990plus", "fused_nasa_archive"]]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "balsas_ab_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print()

    a = summary.iloc[0]
    b = summary.iloc[1]
    diff = {
        "d_CRPS_mean":  b["crps_mean"]   - a["crps_mean"],
        "d_MAE":        b["mae_q50"]     - a["mae_q50"],
        "d_Bias":       b["bias_q50"]    - a["bias_q50"],
        "d_KGE":        b["kge_q50"]     - a["kge_q50"],
        "d_PBIAS_pp":   b["pbias_q50"]   - a["pbias_q50"],
        "d_Cov_90":     b["coverage_90"] - a["coverage_90"],
    }
    print("Fused minus Xavier "
          "(negative is better for CRPS/MAE/|Bias|; positive for KGE/Cov):")
    for k, v in diff.items():
        sign = "+" if v >= 0 else ""
        print(f"  {k:<14s}  {sign}{v:.3f}")


if __name__ == "__main__":
    main()
