"""Re-run the Bayesian state-space NUTS sampler over five soybean cycles.

Existing files ``output/trace_<year>.nc`` were saved by the
climatological forecast pipeline and contain only the Dirichlet
posterior on yearly class weights — they do not include the hydraulic
state-space parameters. This script runs ``fit_model`` from
``bwb.models.water_balance`` (PyMC v5 + NUTS) on five Balsas soybean
cycles 2020-2025 and saves the resulting InferenceData files to
``output/state_space/trace_state_space_<year>.nc``.

The output feeds ``scripts/generate_mcmc_diagnostics.py``, which then
produces the LaTeX table cited in §3.6/§4.2 of the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import arviz as az

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bwb.config.profiles import load_profile  # noqa: E402
from bwb.data.adapters import build_water_balance_inputs, cycle_label  # noqa: E402
from bwb.data.loaders import extract_crop_cycle, load_city_series  # noqa: E402
from bwb.models.water_balance import fit_model, diagnostics_summary  # noqa: E402

CITY = "Balsas"
CYCLES = [2020, 2021, 2022, 2023, 2024]
PROFILE = "matopiba"
OUT = ROOT / "output" / "state_space"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    profile = load_profile(PROFILE)
    df = load_city_series(CITY)

    for cy in CYCLES:
        out_path = OUT / f"trace_state_space_{cy}_{cy + 1}.nc"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        print(f"[run]  {CITY}/{cy}-{cy + 1}: NUTS, 4 chains × 600 draws")
        cycle_df = extract_crop_cycle(df, cy)
        inputs = build_water_balance_inputs(
            cycle_df, profile, city=CITY, season=cycle_label(cy),
        )
        idata, _ = fit_model(
            inputs, draws=600, tune=600, chains=4,
            target_accept=0.95, random_seed=42, progressbar=False,
        )
        diag = diagnostics_summary(idata)
        print("       diagnostics:", diag)
        az.to_netcdf(idata, out_path)
        print(f"       wrote {out_path.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
