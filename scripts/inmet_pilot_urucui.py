"""Pilot validation: INMET A346 (Urucuí, PI) vs NASA / Archive / Fused.

Scope
-----
* Single station — A346, Urucuí/PI (-7.44°, -44.34°, 399 m a.s.l.).
* Single window — 2026-01-01 .. 2026-05-07 (period covered by the
  ``data_raw/inmet/a346.csv`` download).
* Variables compared — Tmax, Tmin, RH, u2, Rs, ETo. **Precipitation is
  excluded** because the INMET CSV ships with all rain values empty
  for this station/period.

Outputs
-------
* ``output/inmet_validation/urucui_daily.csv`` — daily INMET aggregates
  with QC counts and computed ETo.
* ``output/inmet_validation/urucui_compare.csv`` — side-by-side daily
  values from INMET, NASA, Archive, Fused.
* ``output/inmet_validation/urucui_summary.csv`` — bias / RMSE /
  correlation per (variable × source).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bwb.data.sources.climate import (  # noqa: E402
    compute_eto_fao56_pm,
    download_nasa_power,
    download_openmeteo_archive,
    fuse_climate_sources,
    wind_10m_to_2m_fao56,
)

# Station metadata (queried from INMET API)
STATION_CODE = "A346"
CITY = "Urucuí"
LAT, LON, ELEV = -7.44138888, -44.34499999, 398.83
START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-05-07")

INMET_CSV = ROOT / "data_raw" / "inmet" / "a346.csv"
OUT_DIR = ROOT / "output" / "inmet_validation"
CACHE_DIR = OUT_DIR / "_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Minimum hours of valid data required to keep a daily aggregate
MIN_HOURS_TEMP = 20      # Tmax / Tmin / Tmean
MIN_HOURS_RH = 18        # daytime + nighttime
MIN_HOURS_WIND = 20
MIN_HOURS_RAD = 8        # only daylight hours have non-null radiation


# ---------------------------------------------------------------------------
# INMET hourly → daily aggregation
# ---------------------------------------------------------------------------


def _load_inmet_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8", na_values=[""])
    rad_col = next(c for c in df.columns if "Radiacao" in c)
    out = pd.DataFrame({
        "datetime": pd.to_datetime(df["Data"] + " " + df["Hora (UTC)"].astype(str).str.zfill(4),
                                    format="%d/%m/%Y %H%M", utc=True),
        "T":        df["Temp. Ins. (C)"].astype(float),
        "Tmax_h":   df["Temp. Max. (C)"].astype(float),
        "Tmin_h":   df["Temp. Min. (C)"].astype(float),
        "RH":       df["Umi. Ins. (%)"].astype(float),
        "u_10m":    df["Vel. Vento (m/s)"].astype(float),
        "Rs_kJ":    df[rad_col].astype(float),
    })
    return out


def _u10_to_u2(u10: pd.Series) -> pd.Series:
    """Wrap the canonical FAO-56 Eq. 47 helper for use on a Series."""
    return pd.Series(wind_10m_to_2m_fao56(u10.to_numpy()), index=u10.index)


def aggregate_inmet_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    df = df_hourly.copy()
    # Convert UTC → local Urucuí (UTC-3) so the daily window is the
    # local civil day. Without this, "max temp at 18 UTC" leaks into
    # the next civil day.
    df["date_local"] = (df["datetime"] - pd.Timedelta(hours=3)).dt.normalize().dt.tz_localize(None)

    grouped = df.groupby("date_local")
    daily = pd.DataFrame({
        "Tmax":   grouped["Tmax_h"].max(),
        "Tmin":   grouped["Tmin_h"].min(),
        "Tmean":  grouped["T"].mean(),
        "RH":     grouped["RH"].mean(),
        "u_10m":  grouped["u_10m"].mean(),
        # Rs hourly is in kJ/m²; daily total kJ → MJ
        "Rs":     grouped["Rs_kJ"].sum(min_count=1) / 1000.0,
        # QC counters
        "n_hours_T":   grouped["T"].count(),
        "n_hours_RH":  grouped["RH"].count(),
        "n_hours_u":   grouped["u_10m"].count(),
        "n_hours_Rs":  grouped["Rs_kJ"].count(),
    }).reset_index().rename(columns={"date_local": "date"})

    # u2 from u10
    daily["u2"] = _u10_to_u2(daily["u_10m"])
    return daily


def filter_valid_days(daily: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (daily["n_hours_T"]  >= MIN_HOURS_TEMP)
        & (daily["n_hours_RH"] >= MIN_HOURS_RH)
        & (daily["n_hours_u"]  >= MIN_HOURS_WIND)
        & (daily["n_hours_Rs"] >= MIN_HOURS_RAD)
    )
    return daily.loc[keep].reset_index(drop=True)


def add_eto_fao56(daily: pd.DataFrame, *, lat: float, elevation_m: float) -> pd.DataFrame:
    out = daily.copy()
    doy = out["date"].dt.dayofyear.to_numpy()
    out["ETo"] = compute_eto_fao56_pm(
        Tmax=out["Tmax"].to_numpy(), Tmin=out["Tmin"].to_numpy(),
        Tmean=out["Tmean"].to_numpy(), RH=out["RH"].to_numpy(),
        u2=out["u2"].to_numpy(), Rs=out["Rs"].to_numpy(),
        doy=doy, lat_deg=lat, elevation_m=elevation_m,
    )
    return out


# ---------------------------------------------------------------------------
# Alternative sources (cached)
# ---------------------------------------------------------------------------


def _cached(name: str, builder) -> pd.DataFrame:
    p = CACHE_DIR / f"{name}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        return df
    df = builder()
    df.to_parquet(p)
    return df


def fetch_alt_sources(start, end):
    end_archive = min(end, pd.Timestamp.today() - pd.Timedelta(days=2))
    nasa = _cached("nasa_urucui",
                   lambda: download_nasa_power(lat=LAT, lon=LON,
                                                start=start.date(),
                                                end=end_archive.date(),
                                                elevation_m=ELEV))
    arch = _cached("archive_urucui",
                   lambda: download_openmeteo_archive(lat=LAT, lon=LON,
                                                       start=start.date(),
                                                       end=end_archive.date()))
    fused = fuse_climate_sources(
        {"nasa_power": nasa, "openmeteo_archive": arch},
        lat=LAT, elevation_m=ELEV,
    )
    return nasa, arch, fused


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def per_variable_metrics(merged: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for var in ["Tmax", "Tmin", "RH", "u2", "Rs", "ETo"]:
        a = merged[f"{var}_inmet"].to_numpy()
        b = merged[f"{var}_{source}"].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            rows.append({"source": source, "variable": var, "n": int(m.sum())})
            continue
        bias = float(np.mean(b[m] - a[m]))
        rmse = float(np.sqrt(np.mean((b[m] - a[m]) ** 2)))
        if a[m].std() > 1e-9 and b[m].std() > 1e-9:
            r = float(np.corrcoef(a[m], b[m])[0, 1])
        else:
            r = float("nan")
        rows.append({"source": source, "variable": var, "n": int(m.sum()),
                     "bias": bias, "rmse": rmse, "r": r,
                     "inmet_mean": float(a[m].mean()),
                     "src_mean": float(b[m].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"[1/5] reading {INMET_CSV.name}")
    hourly = _load_inmet_hourly(INMET_CSV)
    print(f"      {len(hourly)} hourly records")

    print("[2/5] aggregating to daily (local civil day, UTC-3)")
    daily_raw = aggregate_inmet_daily(hourly)
    daily = filter_valid_days(daily_raw)
    print(f"      kept {len(daily)} / {len(daily_raw)} daily aggregates after QC")
    daily = add_eto_fao56(daily, lat=LAT, elevation_m=ELEV)
    daily.to_csv(OUT_DIR / "urucui_daily.csv", index=False)

    print("[3/5] downloading NASA + Archive (cached) and fusing")
    nasa, arch, fused = fetch_alt_sources(START, END)
    print(f"      nasa: {len(nasa)} | archive: {len(arch)} | fused: {len(fused)}")

    print("[4/5] aligning sources on the INMET-valid date set")
    merged = daily[["date", "Tmax", "Tmin", "RH", "u2", "Rs", "ETo"]].rename(
        columns={c: f"{c}_inmet" for c in ["Tmax", "Tmin", "RH", "u2", "Rs", "ETo"]}
    )
    for label, df in [("nasa", nasa), ("archive", arch), ("fused", fused)]:
        sub = df[["date", "Tmax", "Tmin", "RH", "u2", "Rs", "ETo"]].rename(
            columns={c: f"{c}_{label}" for c in ["Tmax", "Tmin", "RH", "u2", "Rs", "ETo"]}
        )
        merged = merged.merge(sub, on="date", how="inner")
    merged.to_csv(OUT_DIR / "urucui_compare.csv", index=False)
    print(f"      {len(merged)} dates with INMET + all 3 alt sources")

    print("[5/5] computing per-variable bias / RMSE / correlation")
    summary = pd.concat([
        per_variable_metrics(merged, "nasa"),
        per_variable_metrics(merged, "archive"),
        per_variable_metrics(merged, "fused"),
    ], ignore_index=True)
    summary.to_csv(OUT_DIR / "urucui_summary.csv", index=False)

    print()
    print("=" * 96)
    print(f"INMET vs alternative sources — {CITY} ({STATION_CODE}), "
          f"{merged['date'].min().date()}..{merged['date'].max().date()}, "
          f"N = {len(merged)} days")
    print("=" * 96)
    pivot_rmse = summary.pivot(index="variable", columns="source", values="rmse").round(3)
    pivot_bias = summary.pivot(index="variable", columns="source", values="bias").round(3)
    pivot_r    = summary.pivot(index="variable", columns="source", values="r").round(3)
    inmet_mean = summary[summary["source"] == "fused"].set_index("variable")["inmet_mean"].round(2)

    out = pd.DataFrame({
        "INMET_mean": inmet_mean,
        "bias_NASA": pivot_bias["nasa"],   "rmse_NASA": pivot_rmse["nasa"],   "r_NASA": pivot_r["nasa"],
        "bias_Arch": pivot_bias["archive"],"rmse_Arch": pivot_rmse["archive"],"r_Arch": pivot_r["archive"],
        "bias_Fuse": pivot_bias["fused"],  "rmse_Fuse": pivot_rmse["fused"],  "r_Fuse": pivot_r["fused"],
    })
    print(out.to_string())
    print()
    # Where does fusion help vs the better single source?
    print("Per-variable winner by RMSE (best is bold):")
    for var in pivot_rmse.index:
        row = pivot_rmse.loc[var]
        winner = row.idxmin()
        print(f"  {var:5s}  NASA={row['nasa']:.3f}  Archive={row['archive']:.3f}  Fused={row['fused']:.3f}   <- best: {winner}")
    print()
    print(f"Wrote {OUT_DIR/'urucui_daily.csv'}, urucui_compare.csv, urucui_summary.csv")


if __name__ == "__main__":
    main()
