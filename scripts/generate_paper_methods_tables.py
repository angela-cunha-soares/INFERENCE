"""Generate the three Materials & Methods tables for the manuscript.

Outputs in ``output/paper_tables/``:

* ``table_kc_soybean_5stage.tex`` -- 5-stage step Kc curve for 90-day soybean
  (FAO-56 Chapter 6 simplified time-averaged Kc, after Allen et al. 1998 and
  Steduto et al. 2012, FAO-66; Embrapa convention used in Brazilian soybean
  agronomy). Sources the values from ``matopiba.toml``.
* ``table_van_genuchten_matopiba.tex`` -- van Genuchten (1980) retention
  parameters for the latosol profile used as the regional reference in the
  Bayesian theta model. Three depths: 0-20, 20-40, 40-60 cm.
* ``table_soils_matopiba.tex`` -- per-city CAD/AWC from SNIRH/ANA-UFPR
  (Santos et al. TED ANA-UFPR; SNIRH 28fe4baa) for the 10 MATOPIBA cities,
  using Z_r = 60 cm consistent with the 90-day early-cycle soybean.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_TABLES = ROOT / "output" / "paper_tables"
PAPER_TABLES.mkdir(parents=True, exist_ok=True)

if sys.version_info >= (3, 11):
    import tomllib
else:                                                   # pragma: no cover
    import tomli as tomllib                             # type: ignore[no-redef]


def _load_profile(name: str = "matopiba") -> dict:
    p = ROOT / "src" / "bwb" / "config" / "regional" / f"{name}.toml"
    with p.open("rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Table 1 — 5-stage step Kc curve
# ---------------------------------------------------------------------------

def write_table_kc():
    profile = _load_profile()
    crop = profile["crop"]
    rows = [
        ("Initial",       "I",   crop["L_ini"],     crop["kc_ini"],
         "Sowing through 10\\% canopy cover; Kc dominated by soil evaporation."),
        ("Development",   "II",  crop["L_dev"],     crop["kc_dev"],
         "Canopy expansion from 10\\% to effective full cover."),
        ("Mid-season",    "III", crop["L_mid"],     crop["kc_mid"],
         "Flowering through grain fill; Kc at its peak."),
        ("Late-season",   "IV",  crop["L_late"],    crop["kc_late"],
         "Maturation and start of senescence."),
        ("Harvest",       "V",   crop["L_harvest"], crop["kc_harvest"],
         "Pod dry-down through harvest; near-bare-soil evaporation."),
    ]
    total_days = sum(r[2] for r in rows)
    cycle_days = int(crop["cycle_days"])
    if total_days != cycle_days:
        print(f"  [warn] sum(L_*) = {total_days} != cycle_days = {cycle_days}",
              file=sys.stderr)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Crop coefficient (\(K_c\)) curve for the 90-day "
        r"early-cycle soybean grown in MATOPIBA. Stage durations sum to "
        + str(total_days) + r" days; planting on December 1. Each phase has a "
        r"constant \(K_c\) value following the simplified time-averaged "
        r"formulation of FAO-56 Chapter 6 \citep{Allen1998FAO56}, as adopted "
        r"by Brazilian soybean agronomy \citep{EmbrapaSoja, Steduto2012FAO66}.}",
        r"\label{tab:kc-soybean-5stage}",
        r"\begin{tabular}{llrcl}",
        r"\toprule",
        r"Stage & Phase & Days & \(K_c\) & Phenological description \\",
        r"\midrule",
    ]
    for name, phase, L, kc, descr in rows:
        tex.append(f"{name} & {phase} & {L} & {kc:.2f} & {descr} \\\\")
    tex.extend([
        r"\midrule",
        r"\textbf{Total} & & \textbf{" + str(total_days) +
        r"} & \(\bar{K_c} \approx 0.87\) & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    out = PAPER_TABLES / "table_kc_soybean_5stage.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Table 2 — van Genuchten parameters
# ---------------------------------------------------------------------------

def write_table_van_genuchten():
    profile = _load_profile()
    vg = profile["soil"]["van_genuchten"]
    depths = [
        ("0-20",   vg["depth_0_20"]),
        ("20-40",  vg["depth_20_40"]),
        ("40-60",  vg["depth_40_60"]),
    ]
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Components and parameters of the van Genuchten "
        r"\citep{vanGenuchten1980} retention equation for the regional "
        r"reference profile (\textit{Latossolo Vermelho}, sandy phase, "
        r"S\'erie Sert\~aozinho), used as a representative latosol of "
        r"MATOPIBA in the absence of site-specific HYBRAS \citep{Ottoni2018HYBRAS} "
        r"profiles for each of the ten study municipalities. \(\theta_r\) is "
        r"the volumetric residual soil moisture (m\textsuperscript{3}\,m\textsuperscript{-3}); "
        r"\(\theta_s\) is the saturated volumetric soil moisture; \(\alpha\), "
        r"\(m\) and \(n\) are the numerical fitting parameters of the "
        r"equation. Profile parameters by horizon depth.}",
        r"\label{tab:van-genuchten-matopiba}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Depth & \(\theta_s\) & \(\theta_r\) & \(\alpha\) & \(m\) & \(n\) \\",
        r"(cm) & \multicolumn{2}{c}{(m\textsuperscript{3}\,m\textsuperscript{-3})} & "
        r"(kPa\textsuperscript{-1}) & & \\",
        r"\midrule",
    ]
    for name, params in depths:
        tex.append(
            f"{name} & "
            f"{params['theta_s']:.3f} & "
            f"{params['theta_r']:.3f} & "
            f"{params['alpha']:.4f} & "
            f"{params['m']:.3f} & "
            f"{params['n']:.3f} \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    out = PAPER_TABLES / "table_van_genuchten_matopiba.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Table 3 — per-city CAD/AWC
# ---------------------------------------------------------------------------

def write_table_soils_matopiba():
    profile = _load_profile()
    soils = profile["soils"]
    z_r_cm = int(profile["crop"]["root_depth_cm"])

    rows = []
    for city, d in soils.items():
        rows.append((
            city.replace("_", " "),
            int(d["ibge_code"]),
            float(d["cad_m3_m3"]),
            float(d["awc_mm"]),
        ))
    rows.sort(key=lambda r: r[3])    # sort by AWC ascending — drought-prone first

    cad_mean = sum(r[2] for r in rows) / len(rows)
    awc_mean = sum(r[3] for r in rows) / len(rows)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Available water capacity (AWC) for the 10 MATOPIBA "
        r"municipalities used in the validation grid, derived from the "
        r"national CAD dataset of the Brazilian Water Agency "
        r"\citep{SantosANAUFPR_TED}, available at SNIRH "
        r"\citep{SNIRH28fe4baa}. The CAD volumetric value "
        r"(m\textsuperscript{3}\,m\textsuperscript{-3}) is the per-municipality "
        r"average over the 0--120\,cm reference profile, computed by the "
        r"UFPR/ANA team via Saxton \& Rawls "
        r"\citep{SaxtonRawls2006} pedotransfer functions calibrated on "
        r"HYBRAS \citep{Ottoni2018HYBRAS} laboratory profiles and "
        r"extrapolated through the 1:250{,}000 RADAM/IBGE soil map. The "
        r"AWC in mm is integrated over the early-cycle soybean root zone "
        r"\(Z_r=" + str(z_r_cm) + r"\)\,cm: AWC\textsubscript{mm} = "
        r"CAD \(\times Z_r \times 10\). Cities sorted by ascending AWC.}",
        r"\label{tab:soils-matopiba}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Municipality & IBGE code & CAD (m\textsuperscript{3}\,m\textsuperscript{-3}) "
        r"& AWC (mm) \\",
        r"\midrule",
    ]
    for name, code, cad, awc in rows:
        tex.append(f"{name} & {code} & {cad:.4f} & {awc:.1f} \\\\")
    tex.extend([
        r"\midrule",
        f"\\textbf{{Mean}} & & \\textbf{{{cad_mean:.4f}}} & "
        f"\\textbf{{{awc_mean:.1f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    out = PAPER_TABLES / "table_soils_matopiba.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    return out


def main():
    print(f"Output dir: {PAPER_TABLES.relative_to(ROOT)}")
    for fn, label in [
        (write_table_kc,             "Kc 5-stage step (FAO-56 Cap. 6 + FAO-66)"),
        (write_table_van_genuchten,  "van Genuchten retention (Sertaozinho ref)"),
        (write_table_soils_matopiba, "Per-city CAD/AWC (SNIRH/ANA)"),
    ]:
        path = fn()
        print(f"  -> {path.relative_to(ROOT)}  ({label})")
    print("\nDone.")


if __name__ == "__main__":
    main()
