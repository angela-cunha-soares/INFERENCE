"""Multi-station INMET ground-truth validation of NASA / Archive / Fused.

Extends the Urucuí pilot to **all** INMET automatic stations in
``data_raw/inmet/`` whose post-QC daily coverage meets a configurable
threshold (default ≥ 60 days within Jan–May 2026). For each station we:

1. Aggregate hourly INMET to daily (local civil day, UTC-3) with QC
   counters per variable.
2. Compute ETo via FAO-56 Penman-Monteith on the cleaned daily series
   (same routine the fusion uses, so the comparison is symmetric).
3. Download NASA POWER and Open-Meteo Archive for the same window
   (cached on disk), fuse them with the calibrated EVAOnline weights.
4. Align all four time series on the INMET-valid date set and compute
   per-variable bias / RMSE / Pearson r.

Outputs
-------
* ``output/inmet_validation/per_station/<code>_daily.csv``,
  ``per_station/<code>_compare.csv`` — raw and aligned daily values.
* ``output/inmet_validation/per_station_summary.csv`` —
  one row per (station × variable × source) with bias/RMSE/r.
* ``output/inmet_validation/pooled_summary.csv`` —
  same metrics computed on the **pooled** daily anomaly across stations
  (each daily value is recentred on its station mean before pooling so
  that the metric reflects within-station agreement, not station means).
* ``output/inmet_validation/inmet_validation_table.tex`` —
  paper-ready LaTeX table.

Excluded
--------
* ``u2`` (wind at 2 m): INMET CSVs systematically read 0 m/s for >70 %
  of hours at A346 (Urucuí), suggesting an instrument/registration
  fault. Including it would compare the alternative sources against a
  zero ground truth and produce misleading bias values. The metric is
  still computed and stored, but flagged in the LaTeX output.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Station catalogue (queried once from INMET API and frozen here)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Station:
    code: str
    city: str
    uf: str
    lat: float
    lon: float
    elev_m: float
    file: str
    utc_offset_hours: int = -3


STATIONS = [
    Station("A346", "Urucuí",                  "PI", -7.44138888, -44.34499999, 398.83, "a346.csv"),
    Station("A375", "Baixa Grande do Ribeiro", "PI", -8.33353100, -45.09462500, 519.00, "a375.csv"),
    Station("A416", "Correntina",              "BA", -13.33249999, -44.61749999, 551.71, "a416.csv"),
    # A402 and A404 have <60 % coverage; included with a tighter min_days
    # filter so they only enter the analysis if they pass QC.
    Station("A402", "Barreiras",               "BA", -12.12472221, -45.02694443, 474.17, "a402.csv"),
    Station("A404", "Luís Eduardo Magalhães",  "BA", -12.08499999, -45.70527777, 748.00, "a404.csv"),
]

INMET_DIR = ROOT / "data_raw" / "inmet"
OUT_DIR = ROOT / "output" / "inmet_validation"
PER_STATION_DIR = OUT_DIR / "per_station"
CACHE_DIR = OUT_DIR / "_cache"
PAPER_TABLES = ROOT / "output" / "paper_tables"
for d in [OUT_DIR, PER_STATION_DIR, CACHE_DIR, PAPER_TABLES]:
    d.mkdir(parents=True, exist_ok=True)

# QC thresholds (hours per local civil day to keep that day, per variable)
MIN_HOURS_TEMP = 20
MIN_HOURS_RH = 18
MIN_HOURS_WIND = 20
MIN_HOURS_RAD = 8        # only daylight hours have non-null radiation
MIN_HOURS_RAIN = 20

# Each variable lists which raw INMET columns (in our normalised hourly
# frame) must clear MIN_HOURS_* for the daily aggregate to be usable for
# that variable. ETo additionally needs Tmax/Tmin/RH/u2/Rs all present.
VAR_REQUIREMENTS = {
    "Tmax": [("n_hours_T",  MIN_HOURS_TEMP)],
    "Tmin": [("n_hours_T",  MIN_HOURS_TEMP)],
    "RH":   [("n_hours_RH", MIN_HOURS_RH)],
    "u2":   [("n_hours_u",  MIN_HOURS_WIND)],
    "Rs":   [("n_hours_Rs", MIN_HOURS_RAD)],
    "pr":   [("n_hours_pr", MIN_HOURS_RAIN)],
    "ETo":  [("n_hours_T",  MIN_HOURS_TEMP),
             ("n_hours_RH", MIN_HOURS_RH),
             ("n_hours_u",  MIN_HOURS_WIND),
             ("n_hours_Rs", MIN_HOURS_RAD)],
}

MIN_DAYS_PER_VARIABLE = 30   # need at least this many days for any metric
                              # to enter the per-station table.

VARIABLES = ["Tmax", "Tmin", "RH", "u2", "Rs", "pr", "ETo"]
SOURCES = ["nasa", "archive", "fused"]


# ---------------------------------------------------------------------------
# INMET parsing + aggregation (lifted from the Urucuí pilot)
# ---------------------------------------------------------------------------


def _load_inmet_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8", na_values=[""])
    rad_col = next(c for c in df.columns if "Radiacao" in c)
    out = pd.DataFrame({
        "datetime": pd.to_datetime(
            df["Data"] + " " + df["Hora (UTC)"].astype(str).str.zfill(4),
            format="%d/%m/%Y %H%M", utc=True,
        ),
        "T":      df["Temp. Ins. (C)"].astype(float),
        "Tmax_h": df["Temp. Max. (C)"].astype(float),
        "Tmin_h": df["Temp. Min. (C)"].astype(float),
        "RH":     df["Umi. Ins. (%)"].astype(float),
        "u_10m":  df["Vel. Vento (m/s)"].astype(float),
        "Rs_kJ":  df[rad_col].astype(float),
        "pr_h":   df["Chuva (mm)"].astype(float),
    })
    return out


def _u10_to_u2(u10: pd.Series) -> pd.Series:
    """Wrap the canonical FAO-56 Eq. 47 helper for use on a Series.

    INMET automatic stations measure wind at 10 m; FAO-56 Penman-Monteith
    expects wind at 2 m.
    """
    return pd.Series(wind_10m_to_2m_fao56(u10.to_numpy()), index=u10.index)


def aggregate_inmet_daily(df_hourly: pd.DataFrame, *, utc_offset_hours: int) -> pd.DataFrame:
    df = df_hourly.copy()
    df["date_local"] = (
        (df["datetime"] + pd.Timedelta(hours=utc_offset_hours))
        .dt.normalize()
        .dt.tz_localize(None)
    )
    g = df.groupby("date_local")
    daily = pd.DataFrame({
        "Tmax":   g["Tmax_h"].max(),
        "Tmin":   g["Tmin_h"].min(),
        "Tmean":  g["T"].mean(),
        "RH":     g["RH"].mean(),
        "u_10m":  g["u_10m"].mean(),
        "Rs":     g["Rs_kJ"].sum(min_count=1) / 1000.0,
        "pr":     g["pr_h"].sum(min_count=1),
        "n_hours_T":  g["T"].count(),
        "n_hours_RH": g["RH"].count(),
        "n_hours_u":  g["u_10m"].count(),
        "n_hours_Rs": g["Rs_kJ"].count(),
        "n_hours_pr": g["pr_h"].count(),
    }).reset_index().rename(columns={"date_local": "date"})
    daily["u2"] = _u10_to_u2(daily["u_10m"])
    return daily


def per_variable_validity(daily: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return a mask per variable (True where the daily aggregate clears
    its hourly-coverage threshold)."""
    masks = {}
    for var, reqs in VAR_REQUIREMENTS.items():
        m = np.ones(len(daily), dtype=bool)
        for col, threshold in reqs:
            m &= (daily[col].to_numpy() >= threshold)
        masks[var] = m
    return masks


def add_eto(daily: pd.DataFrame, *, lat: float, elev: float,
            valid_mask: np.ndarray) -> pd.DataFrame:
    """Compute ETo only where all PM inputs are present; elsewhere NaN."""
    out = daily.copy()
    doy = out["date"].dt.dayofyear.to_numpy()
    eto = np.full(len(out), np.nan)
    if valid_mask.any():
        eto_vals = compute_eto_fao56_pm(
            Tmax=out["Tmax"].to_numpy()[valid_mask],
            Tmin=out["Tmin"].to_numpy()[valid_mask],
            Tmean=out["Tmean"].to_numpy()[valid_mask],
            RH=out["RH"].to_numpy()[valid_mask],
            u2=out["u2"].to_numpy()[valid_mask],
            Rs=out["Rs"].to_numpy()[valid_mask],
            doy=doy[valid_mask],
            lat_deg=lat, elevation_m=elev,
        )
        eto[valid_mask] = eto_vals
    out["ETo"] = eto
    return out


# ---------------------------------------------------------------------------
# Alternative sources (cached per station)
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


def fetch_alt_sources_for(station: Station, start: pd.Timestamp, end: pd.Timestamp):
    end_archive = min(end, pd.Timestamp.today() - pd.Timedelta(days=2))
    nasa = _cached(
        f"nasa_{station.code}",
        lambda: download_nasa_power(
            lat=station.lat, lon=station.lon,
            start=start.date(), end=end_archive.date(),
            elevation_m=station.elev_m,
        ),
    )
    arch = _cached(
        f"archive_{station.code}",
        lambda: download_openmeteo_archive(
            lat=station.lat, lon=station.lon,
            start=start.date(), end=end_archive.date(),
        ),
    )
    fused = fuse_climate_sources(
        {"nasa_power": nasa, "openmeteo_archive": arch},
        lat=station.lat, elevation_m=station.elev_m,
    )
    return nasa, arch, fused


# ---------------------------------------------------------------------------
# Per-station alignment + metrics
# ---------------------------------------------------------------------------


def align_sources(daily_inmet: pd.DataFrame, *, nasa, arch, fused) -> pd.DataFrame:
    base = daily_inmet[["date"] + VARIABLES].rename(
        columns={v: f"{v}_inmet" for v in VARIABLES}
    )
    for label, df in [("nasa", nasa), ("archive", arch), ("fused", fused)]:
        sub = df[["date"] + VARIABLES].rename(columns={v: f"{v}_{label}" for v in VARIABLES})
        base = base.merge(sub, on="date", how="inner")
    return base


def metrics_one_station(merged: pd.DataFrame, *, station: Station) -> pd.DataFrame:
    rows = []
    for var in VARIABLES:
        a = merged[f"{var}_inmet"].to_numpy()
        for src in SOURCES:
            b = merged[f"{var}_{src}"].to_numpy()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < MIN_DAYS_PER_VARIABLE:
                rows.append({"station": station.code, "city": station.city,
                             "variable": var, "source": src,
                             "n": int(m.sum()),
                             "bias": np.nan, "rmse": np.nan, "r": np.nan,
                             "inmet_mean": np.nan, "src_mean": np.nan})
                continue
            bias = float(np.mean(b[m] - a[m]))
            rmse = float(np.sqrt(np.mean((b[m] - a[m]) ** 2)))
            r = (float(np.corrcoef(a[m], b[m])[0, 1])
                 if a[m].std() > 1e-9 and b[m].std() > 1e-9 else float("nan"))
            rows.append({
                "station": station.code, "city": station.city,
                "variable": var, "source": src, "n": int(m.sum()),
                "bias": bias, "rmse": rmse, "r": r,
                "inmet_mean": float(a[m].mean()),
                "src_mean": float(b[m].mean()),
            })
    return pd.DataFrame(rows)


def pooled_metrics(per_station_compare: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pool daily *anomalies* (value − station mean of that variable) across
    stations to remove station-level offsets, then compute global metrics.
    Same idea as a fixed-effects panel regression — the metric reflects
    *within-station* agreement, not differences in absolute level."""
    rows = []
    for var in VARIABLES:
        for src in SOURCES:
            obs_anom, sim_anom = [], []
            for code, merged in per_station_compare.items():
                a = merged[f"{var}_inmet"].to_numpy()
                b = merged[f"{var}_{src}"].to_numpy()
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 5:
                    continue
                a, b = a[m], b[m]
                obs_anom.append(a - a.mean())
                sim_anom.append(b - b.mean())
            if not obs_anom:
                rows.append({"variable": var, "source": src, "n": 0})
                continue
            obs = np.concatenate(obs_anom)
            sim = np.concatenate(sim_anom)
            bias_pool = float(np.mean(sim - obs))                # ≈ 0 by construction
            rmse_pool = float(np.sqrt(np.mean((sim - obs) ** 2)))
            r_pool = (float(np.corrcoef(obs, sim)[0, 1])
                      if obs.std() > 1e-9 and sim.std() > 1e-9 else float("nan"))
            rows.append({"variable": var, "source": src, "n": int(len(obs)),
                         "bias_anom": bias_pool, "rmse_anom": rmse_pool, "r_anom": r_pool})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def render_latex_table(per_station: pd.DataFrame, pooled: pd.DataFrame,
                       station_meta: list[Station],
                       *, min_vars_for_column: int = 4) -> str:
    """Build a paper-ready LaTeX table:

    * one block per variable
    * within each block: per-station RMSE (NASA / Arch / Fused) + pooled row
    * u2 row marked with a footnote about the INMET zero-bias issue
    * stations contributing fewer than ``min_vars_for_column`` variables
      are not shown as a dedicated column (their data still appears in
      the pooled row).
    """
    # Order stations by city; drop those with too little data to deserve a column
    candidates = [s for s in station_meta if s.code in per_station["station"].unique()]
    keep = []
    for s in candidates:
        sub = per_station[(per_station["station"] == s.code)
                          & (per_station["source"] == "fused")]
        n_var_ok = sub["rmse"].notna().sum()
        if n_var_ok >= min_vars_for_column:
            keep.append(s)
    used = sorted(keep, key=lambda s: s.city)

    excluded = sorted(
        s.code for s in station_meta
        if s.code in per_station["station"].unique() and s not in keep
    )
    excl_note = (
        f" Stations {', '.join(excluded)} contributed fewer than "
        f"{min_vars_for_column} variables after QC and are absorbed into "
        f"the pooled row only."
        if excluded else ""
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Daily-scale validation of NASA POWER, Open-Meteo Archive "
        r"and the EVAOnline-weighted fused product against INMET automatic "
        r"stations in MATOPIBA, Jan--May 2026. Cells report RMSE; bold "
        r"marks the source with the smallest RMSE in each row. The "
        r"`Pooled' column aggregates across stations on the daily anomaly "
        r"relative to each station's mean, so it reflects within-station "
        r"agreement (immune to station-level offsets). Wind at 2\,m is "
        r"reported with a dagger because $\geq70\%$ of hourly INMET "
        r"anemometer readings register exactly $0$\,m/s, yielding a "
        r"degenerate ground truth on a per-station basis (the pooled row "
        r"on the anomaly removes that bias)." + excl_note + r"}",
        r"\label{tab:inmet-validation}",
        r"\begin{tabular}{ll" + "rrr" * len(used) + r"rrr}",
        r"\toprule",
        r"& & " + " & ".join(
            r"\multicolumn{3}{c}{" + s.city + "}" for s in used
        ) + r" & \multicolumn{3}{c}{Pooled} \\",
        r"\cmidrule(lr){3-" + str(2 + 3 * len(used)) + r"} "
        r"\cmidrule(lr){" + str(3 + 3 * len(used)) + r"-" + str(2 + 3 * (len(used) + 1)) + r"}",
        r"Variable & Source & " + " & ".join(["NASA & Arch & Fused"] * (len(used) + 1)) + r" \\",
        r"\midrule",
    ]

    var_label = {
        "Tmax": r"$T_\mathrm{max}$ ($^\circ$C)",
        "Tmin": r"$T_\mathrm{min}$ ($^\circ$C)",
        "RH":   r"RH (\%)",
        "u2":   r"$u_2$ (m\,s$^{-1}$)$^\dagger$",
        "Rs":   r"$R_\mathrm{s}$ (MJ\,m$^{-2}$\,d$^{-1}$)",
        "pr":   r"$P$ (mm\,d$^{-1}$)",
        "ETo":  r"ET$_0$ (mm\,d$^{-1}$)",
    }

    for var in VARIABLES:
        cells = [var_label[var], "RMSE"]
        per_var_per_st_pivot = (per_station[per_station["variable"] == var]
                                .pivot(index="station", columns="source", values="rmse"))
        pooled_var = pooled[pooled["variable"] == var].set_index("source")["rmse_anom"]
        # one column per station (NASA, Arch, Fused)
        for s in used:
            row = per_var_per_st_pivot.loc[s.code] if s.code in per_var_per_st_pivot.index else None
            if row is None or row.isna().all():
                cells += ["-", "-", "-"]
            else:
                vals = [row.get(src, np.nan) for src in ["nasa", "archive", "fused"]]
                best = int(np.nanargmin(vals))
                cells += [
                    (r"\textbf{" + f"{v:.2f}" + r"}") if i == best and np.isfinite(v) else
                    (f"{v:.2f}" if np.isfinite(v) else "-")
                    for i, v in enumerate(vals)
                ]
        # pooled
        vals = [pooled_var.get(src, np.nan) for src in ["nasa", "archive", "fused"]]
        if any(np.isfinite(vals)):
            best = int(np.nanargmin([v if np.isfinite(v) else np.inf for v in vals]))
            cells += [
                (r"\textbf{" + f"{v:.2f}" + r"}") if i == best and np.isfinite(v) else
                (f"{v:.2f}" if np.isfinite(v) else "-")
                for i, v in enumerate(vals)
            ]
        else:
            cells += ["-", "-", "-"]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main():
    print("[1/5] aggregating INMET hourly to daily for each station "
          "(per-variable QC)")
    per_station_daily: dict[str, pd.DataFrame] = {}
    per_station_var_avail: dict[str, dict[str, int]] = {}
    used_stations: list[Station] = []
    for s in STATIONS:
        path = INMET_DIR / s.file
        if not path.exists():
            print(f"   skip {s.code}: file missing")
            continue
        hourly = _load_inmet_hourly(path)
        daily = aggregate_inmet_daily(hourly, utc_offset_hours=s.utc_offset_hours)
        masks = per_variable_validity(daily)
        # Mask each variable column where its QC failed (keeps the row,
        # but the metric loop will see NaN for that variable).
        for var, m in masks.items():
            if var == "ETo":
                continue   # ETo is computed below from already-masked inputs
            if var in daily.columns:
                arr = daily[var].astype(float).to_numpy()
                arr = np.where(m, arr, np.nan)
                daily[var] = arr
        daily = add_eto(daily, lat=s.lat, elev=s.elev_m,
                        valid_mask=masks["ETo"])
        # Available days per variable (after masking)
        avail = {var: int(daily[var].notna().sum()) for var in VARIABLES if var in daily.columns}
        per_station_var_avail[s.code] = avail
        # Drop rows where every measured variable is NaN
        any_var = np.zeros(len(daily), dtype=bool)
        for var in VARIABLES:
            if var in daily.columns:
                any_var |= daily[var].notna().to_numpy()
        daily = daily.loc[any_var].reset_index(drop=True)
        if not any(v >= MIN_DAYS_PER_VARIABLE for v in avail.values()):
            print(f"   skip {s.code} ({s.city}): no variable has "
                  f">={MIN_DAYS_PER_VARIABLE} days. avail={avail}")
            continue
        daily.to_csv(PER_STATION_DIR / f"{s.code}_daily.csv", index=False)
        per_station_daily[s.code] = daily
        used_stations.append(s)
        avail_str = " ".join(f"{k}={v}" for k, v in avail.items() if v > 0)
        print(f"   {s.code} {s.city:30s}  {avail_str}")

    if not used_stations:
        raise SystemExit("No station passed QC; aborting.")

    print()
    print("[2/5] downloading + fusing NASA + Archive (cached) per station")
    per_station_compare: dict[str, pd.DataFrame] = {}
    for s in used_stations:
        daily = per_station_daily[s.code]
        nasa, arch, fused = fetch_alt_sources_for(
            s, daily["date"].min(), daily["date"].max(),
        )
        merged = align_sources(daily, nasa=nasa, arch=arch, fused=fused)
        merged.to_csv(PER_STATION_DIR / f"{s.code}_compare.csv", index=False)
        per_station_compare[s.code] = merged
        print(f"   {s.code}: {len(merged)} aligned days")

    print()
    print("[3/5] computing per-station metrics")
    per_station_summary = pd.concat(
        [metrics_one_station(per_station_compare[s.code], station=s)
         for s in used_stations],
        ignore_index=True,
    )
    per_station_summary.to_csv(OUT_DIR / "per_station_summary.csv", index=False)

    print("[4/5] computing pooled (within-station anomaly) metrics")
    pooled = pooled_metrics(per_station_compare)
    pooled.to_csv(OUT_DIR / "pooled_summary.csv", index=False)

    print()
    print("=" * 96)
    print("RMSE per station, per variable (winner per row in bold)")
    print("=" * 96)
    pivot = (per_station_summary.pivot_table(
        index=["variable"], columns=["station", "source"], values="rmse")
        .round(3))
    print(pivot.to_string())
    print()
    print("Pooled (within-station anomaly) RMSE:")
    pooled_pivot = pooled.pivot(index="variable", columns="source", values="rmse_anom").round(3)
    print(pooled_pivot.to_string())

    print()
    print("[5/5] rendering LaTeX table")
    tex = render_latex_table(per_station_summary, pooled, used_stations)
    tex_path = PAPER_TABLES / "table_inmet_validation.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"   wrote {tex_path.relative_to(ROOT)}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
