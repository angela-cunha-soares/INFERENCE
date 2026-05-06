"""Run the full Bayesian water-balance validation pipeline.

Examples
--------
# Default: 10 cities x 3 depths x 5 cycles, sequential
python scripts/run_validation.py

# Fast smoke test (1 city, 1 depth, 1 cycle, short chains)
python scripts/run_validation.py --fast --cities Balsas --depths 60 --cycles 2023

# Parallel execution (joblib required)
python scripts/run_validation.py --jobs 4

Outputs go to ``output/validation/`` by default:
    validation_<TS>.csv          per-combination metrics
    validation_<TS>.parquet      same, parquet
    validation_<TS>_manifest.json metadata + counts
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cities", nargs="+", default=None,
                   help="Subset of cities to validate (default: all 10 MATOPIBA cities)")
    p.add_argument("--depths", nargs="+", type=int, default=None,
                   help="Soil depths in cm (default: 40 60 80)")
    p.add_argument("--cycles", nargs="+", type=int, default=None,
                   help="Crop cycle start years (default: 2020 2021 2022 2023 2024)")
    p.add_argument("--profile", default="matopiba",
                   help="Regional profile name (default: matopiba)")
    p.add_argument("--output-dir", default="output/validation",
                   help="Directory for results (default: output/validation)")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel jobs (requires joblib; default: 1 = sequential)")
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: short chains, fewer draws (smoke-test)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from bwb.validation.pipeline import (
        create_validation_config,
        generate_validation_grid,
        generate_validation_report,
        run_validation_pipeline,
        save_validation_results,
    )

    config = create_validation_config(
        output_dir=Path(args.output_dir),
        cities=args.cities,
        soil_depths=args.depths,
        crop_cycles=args.cycles,
        profile_name=args.profile,
        fast_mode=args.fast,
    )

    grid = generate_validation_grid(config)
    print("=" * 72)
    print(" Bayesian water-balance validation")
    print("=" * 72)
    print(f" Profile           : {config.profile_name}")
    print(f" Cities            : {len(config.cities)}")
    print(f" Soil depths (cm)  : {config.soil_depths}")
    print(f" Crop cycles       : {config.crop_cycles}")
    print(f" Total combinations: {len(grid)}")
    print(f" MCMC              : draws={config.draws} tune={config.tune} chains={config.chains}")
    print(f" Output dir        : {config.output_dir}")
    print(f" Parallel jobs     : {args.jobs}")
    print(f" Fast mode         : {config.fast_mode}")
    print()

    t0 = time.time()
    results = run_validation_pipeline(
        config, parallel=args.jobs > 1, n_jobs=args.jobs,
    )
    elapsed = time.time() - t0

    manifest = save_validation_results(results, config.output_dir)
    print()
    print(generate_validation_report(results))
    print()
    print(f"Wall time         : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"CSV               : {manifest['csv']}")
    print(f"Manifest          : {manifest['csv'].replace('.csv', '_manifest.json')}")

    n_fail = int((results.get("status") != "ok").sum()) if "status" in results.columns else 0
    return 1 if n_fail > 0 and not config.fast_mode else 0


if __name__ == "__main__":
    sys.exit(main())
