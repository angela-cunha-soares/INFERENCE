"""Global sensitivity analysis (Sobol' indices) for the water-balance model.

The model output of interest is the seasonal water deficit (sum_t max(ETc - P, 0))
from the deterministic FAO-56 baseline run with the sampled parameters. This is
fast to compute, so we can afford 1000+ samples for stable Sobol indices.

Examples
--------
python scripts/run_sensitivity.py
python scripts/run_sensitivity.py --city Balsas --cycle 2023 --n 1500 --output output/sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", default="Balsas")
    p.add_argument("--cycle", type=int, default=2023)
    p.add_argument("--profile", default="matopiba")
    p.add_argument("--n", type=int, default=1000, help="Saltelli base sample size")
    p.add_argument("--output", default="output/sensitivity",
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from bwb.config.profiles import load_profile
    from bwb.data.adapters import build_kc_curve, run_deterministic_baseline
    from bwb.data.loaders import extract_crop_cycle, load_city_series
    from bwb.validation.sensitivity import (
        SensitivityConfig, compute_sobol_indices, generate_saltelli_samples,
        save_sensitivity_report,
    )

    profile = load_profile(args.profile)
    df = load_city_series(args.city)
    cycle_df = extract_crop_cycle(
        df, args.cycle,
        planting_month=int(profile["crop"]["planting_month"]),
        planting_day=int(profile["crop"]["planting_day"]),
        cycle_days=int(profile["crop"]["cycle_days"]),
    )
    P = cycle_df["pr"].to_numpy(dtype=float)
    ETo = cycle_df["ETo"].to_numpy(dtype=float)
    Kc = build_kc_curve(profile, n_days=len(P))
    z_r_m = float(profile["crop"]["root_depth_cm"]) / 100.0

    def water_deficit(params: dict) -> float:
        traj = run_deterministic_baseline(
            p_daily=P,
            eto_daily=ETo,
            kc_curve=Kc * params["Kc_mult"],
            theta_s=params["theta_s"],
            theta_r=params["theta_r"],
            theta_init=params["theta_init"],
            z_r_m=z_r_m,
        )
        # Deficit proxy: cumulative depletion from saturation across the season
        # (always positive, captures sensitivity even on wet cycles)
        return float(np.sum(np.maximum(params["theta_s"] - traj, 0.0)))

    config = SensitivityConfig(
        n_samples=args.n,
        n_bootstrap=0,
        parameter_ranges={
            "theta_s":    (0.35, 0.50),
            "theta_r":    (0.05, 0.15),
            "Kc_mult":    (0.85, 1.15),
            "theta_init": (0.20, 0.35),
        },
        output_var="seasonal_deficit",
    )

    print("=" * 72)
    print(" Sobol' sensitivity analysis")
    print("=" * 72)
    print(f" City              : {args.city}")
    print(f" Cycle             : {args.cycle}/{args.cycle + 1}")
    print(f" Output target     : {config.output_var}")
    print(f" Saltelli n        : {config.n_samples}")
    print(f" Total model runs  : {(2 + len(config.parameter_ranges)) * config.n_samples}")
    print()

    t0 = time.time()
    samples = generate_saltelli_samples(config, random_seed=args.seed)
    outputs = np.array([water_deficit(row.to_dict()) for _, row in samples.iterrows()])
    indices = compute_sobol_indices(samples, outputs, config)
    elapsed = time.time() - t0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"sobol_{args.city}_{args.cycle}.csv"
    save_sensitivity_report({"indices": indices}, csv_path)

    summary = {
        "city": args.city,
        "cycle": args.cycle,
        "n_samples": config.n_samples,
        "var_y": indices["var_y"],
        "first_order": indices["first_order"],
        "total_order": indices["total_order"],
        "elapsed_s": elapsed,
    }
    (out_dir / f"sobol_{args.city}_{args.cycle}.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"{'parameter':<14s} {'first_order':>12s} {'total_order':>12s}")
    for p in indices["first_order"]:
        s1 = indices["first_order"][p]
        st = indices["total_order"][p]
        print(f"{p:<14s} {s1:>12.4f} {st:>12.4f}")

    print()
    print(f"Wall time   : {elapsed:.1f}s")
    print(f"CSV         : {csv_path}")
    print(f"JSON        : {out_dir / f'sobol_{args.city}_{args.cycle}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
