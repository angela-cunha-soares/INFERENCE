"""bwb command-line interface.

Subcommands
-----------
    info                : print settings + active profile
    posterior-recovery  : posterior-recovery test of the Bayesian model
                          against the FAO-56 deterministic baseline
                          (legacy name: validate)
    validate            : alias of posterior-recovery (deprecated)
    forecast-sequential : climatological forecast across consecutive cycles
                          with sequential Dirichlet-Multinomial update.
                          THIS is the production validation against Xavier.
    sensitivity         : run Sobol' sensitivity analysis
    forecast            : single-cycle Bayesian fit + ensemble propagation
                          (synthetic GEFS-like ensemble)
    figures             : regenerate publication figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bwb import __version__


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    from bwb.config.profiles import load_profile, list_profiles
    from bwb.config.settings import get_settings
    from bwb.data.loaders import list_available_cities

    s = get_settings()
    print("=" * 72)
    print(f" bwb {__version__}")
    print("=" * 72)
    print(f" Project root  : {s.project_root}")
    print(f" Data dir      : {s.data_dir}")
    print(f" Output dir    : {s.output_dir}")
    print(f" Active region : {s.region}")
    print(f" MCMC defaults : draws={s.draws} tune={s.tune} chains={s.chains}")
    print()
    print(f" Profiles available    : {list_profiles()}")
    try:
        cities = list_available_cities()
        print(f" City series available : {len(cities)}")
        for c in cities:
            print(f"   - {c}")
    except Exception as e:
        print(f" City series listing failed: {e}")

    if args.profile:
        try:
            profile = load_profile(args.profile)
            print()
            print(f"Profile {args.profile!r} crop block:")
            for k, v in profile.get("crop", {}).items():
                print(f"  {k:<20s} {v}")
        except Exception as e:
            print(f" load_profile({args.profile!r}) failed: {e}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if getattr(args, "_command_name", None) == "validate":
        print(
            "[deprecation] 'bwb validate' has been renamed to "
            "'bwb posterior-recovery'. This is a posterior-recovery test, "
            "not a true validation. For predictive validation against Xavier, "
            "use 'bwb forecast-sequential'.",
            file=sys.stderr,
        )
    from bwb.validation.pipeline import (
        create_validation_config,
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
    results = run_validation_pipeline(config, parallel=args.jobs > 1, n_jobs=args.jobs)
    manifest = save_validation_results(results, config.output_dir)
    print(generate_validation_report(results))
    print(f"\nCSV: {manifest['csv']}")
    n_fail = int((results.get("status") != "ok").sum()) if "status" in results.columns else 0
    return 1 if n_fail > 0 and not args.fast else 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    # Defer to the orchestration script for the heavy lifting
    from runpy import run_path
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_sensitivity.py"
    sys.argv = [
        "run_sensitivity.py",
        "--city", args.city,
        "--cycle", str(args.cycle),
        "--profile", args.profile,
        "--n", str(args.n),
        "--output", args.output,
        "--seed", str(args.seed),
    ]
    run_path(str(script), run_name="__main__")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    """Fit Bayesian model + propagate a synthetic ensemble for a single cycle."""
    import numpy as np
    from bwb.config.profiles import load_profile
    from bwb.data.adapters import build_water_balance_inputs, cycle_label
    from bwb.data.loaders import extract_crop_cycle, load_city_series
    from bwb.forecast.ensemble import (
        decision_from_ensemble, ensemble_quantiles,
        posterior_from_idata, propagate_ensemble, synthetic_perturbed_ensemble,
    )
    from bwb.models.water_balance import fit_model, diagnostics_summary
    from bwb.phenology.kc_curves import soybean_kc_90d

    profile = load_profile(args.profile)
    df = load_city_series(args.city)
    cycle_df = extract_crop_cycle(df, args.cycle)
    inputs = build_water_balance_inputs(
        cycle_df, profile, city=args.city, season=cycle_label(args.cycle),
    )

    print(f"Fitting Bayesian model for {args.city}/{cycle_label(args.cycle)} ...")
    idata, _ = fit_model(
        inputs, draws=args.draws, tune=args.tune, chains=args.chains,
        progressbar=False,
    )
    diag = diagnostics_summary(idata)
    print("  diagnostics:", diag)

    print("Propagating synthetic ensemble through posterior ...")
    P_ens, ETo_ens = synthetic_perturbed_ensemble(
        inputs.P_daily, inputs.ETo_daily, n_members=args.members,
        random_seed=args.seed,
    )
    posterior = posterior_from_idata(idata)
    traj = propagate_ensemble(
        P_ens, ETo_ens, soybean_kc_90d(), posterior,
        n_posterior_draws=args.posterior_draws, random_seed=args.seed,
    )
    qs = ensemble_quantiles(traj)
    dec = decision_from_ensemble(traj, theta_crit=inputs.soil_params["theta_r"] + 0.02)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({
        "dia_ciclo": np.arange(1, traj.shape[0] + 1),
        "theta_q05": qs["q05"], "theta_q50": qs["q50"], "theta_q95": qs["q95"],
        "p_deficit": dec["p_deficit"], "irrigate": dec["irrigate"],
    }).to_csv(out_dir / f"forecast_{args.city}_{args.cycle}.csv", index=False)
    print(f"  wrote {out_dir / f'forecast_{args.city}_{args.cycle}.csv'}")
    return 0


def cmd_forecast_sequential(args: argparse.Namespace) -> int:
    """Climatological Bayesian forecast across consecutive crop cycles.

    The model is trained on 1961..(target - 1) climatology, forecasts each
    target cycle by Monte-Carlo resampling, then conjugately updates the
    Dirichlet posterior with the observed class before forecasting the
    next cycle.
    """
    import json
    import time

    import numpy as np

    from bwb.config.profiles import load_profile
    from bwb.data.loaders import list_available_cities, load_city_series
    from bwb.forecast.climatological import (
        run_sequential_forecast, save_sequential_outputs,
    )

    profile = load_profile(args.profile)
    cities = args.cities or list_available_cities()
    target_years = args.cycles
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.alpha_init_from and Path(args.alpha_init_from).exists():
        alpha_init = np.array(json.loads(Path(args.alpha_init_from).read_text()))
        print(f"  Initial alpha loaded from {args.alpha_init_from}: {alpha_init.tolist()}")
    else:
        alpha_init = np.ones(3, dtype=float)

    print("=" * 72)
    print(" Climatological Bayesian forecast (sequential)")
    print("=" * 72)
    print(f" Profile          : {args.profile}")
    print(f" Cities           : {len(cities)} -> {cities[:3]}{'...' if len(cities) > 3 else ''}")
    print(f" Target cycles    : {target_years}")
    print(f" Classification   : {args.method}")
    print(f" Simulations/cycle: {args.n_sim}")
    print(f" Initial alpha    : {alpha_init.tolist()}")
    print(f" Output dir       : {out_dir}")
    print()

    overall_summary = []
    t0 = time.time()

    for city in cities:
        print(f"--- {city} ---")
        df = load_city_series(city)
        result = run_sequential_forecast(
            df,
            target_years=target_years,
            profile=profile,
            city=city,
            alpha_init=alpha_init,
            method=args.method,
            n_simulations=args.n_sim,
            random_seed=args.seed,
        )
        manifest = save_sequential_outputs(result, out_dir, profile)

        for year, ev in result.evaluations.items():
            row = {
                "city": city,
                "cycle": ev.season_label,
                "observed_class": ev.observed_class,
                **{f"det_{k}": v for k, v in ev.metrics_deterministic.items()},
                **{f"prob_{k}": v for k, v in ev.metrics_probabilistic.items()},
                **{f"baseline_{k}": v for k, v in ev.metrics_baseline.items()},
                **ev.crpss,
            }
            overall_summary.append(row)
            crpss_naive = ev.crpss.get("CRPSS_vs_naive_climatology", float("nan"))
            print(f"  {ev.season_label}: I_obs={ev.metrics_deterministic['I_total_obs_mm']:.1f} "
                  f"I_q50={ev.metrics_probabilistic['I_total_q50']:.1f} "
                  f"CRPS={ev.metrics_probabilistic['CRPS_I_total_mm']:.2f} "
                  f"cov_SW={ev.metrics_probabilistic['coverage_90_SW_daily']:.2f} "
                  f"CRPSS_naive={crpss_naive:.3f}")

        # Persist final posterior so the operational forecast (2025/26)
        # can pick up where this run left off
        final_alpha = result.final_posterior.alpha.tolist()
        (out_dir / f"alpha_final_{city}.json").write_text(json.dumps(final_alpha))
        print(f"  final alpha = {final_alpha}")
        print(f"  outputs: {len(manifest['per_cycle'])} per-cycle CSVs + summary")
        print()

    if overall_summary:
        import pandas as pd
        summary_df = pd.DataFrame(overall_summary)
        summary_path = out_dir / "sequential_summary_all_cities.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Aggregate summary: {summary_path}")
        print()
        # Headline numbers across all cities x cycles
        for col in ("prob_CRPS_I_total_mm", "prob_coverage_90_SW_daily",
                    "det_KGE_SW", "det_NSE_SW",
                    "CRPSS_vs_naive_climatology",
                    "CRPSS_vs_persistence",
                    "CRPSS_vs_long_term_mean"):
            if col in summary_df.columns:
                vals = summary_df[col].dropna()
                if len(vals):
                    print(f"  {col:35s}  mean={vals.mean():7.3f}  "
                          f"median={vals.median():7.3f}  std={vals.std():7.3f}")

    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from runpy import run_path
    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_paper_figures.py"
    sys.argv = ["generate_paper_figures.py", "--output", args.output, "--results", args.results]
    if args.no_pdf:
        sys.argv.append("--no-pdf")
    run_path(str(script), run_name="__main__")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bwb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"bwb {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("info", help="Print settings and active profile")
    sp.add_argument("--profile", default="matopiba")
    sp.set_defaults(func=cmd_info)

    # New canonical name for the posterior-recovery test
    for name, help_text in [
        ("posterior-recovery",
         "Posterior-recovery test of the Bayesian model against the FAO-56 baseline"),
        ("validate",
         "[deprecated] alias of posterior-recovery"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--cities", nargs="+", default=None)
        sp.add_argument("--depths", nargs="+", type=int, default=None)
        sp.add_argument("--cycles", nargs="+", type=int, default=None)
        sp.add_argument("--profile", default="matopiba")
        sp.add_argument("--output-dir", default="output/posterior_recovery")
        sp.add_argument("--jobs", type=int, default=1)
        sp.add_argument("--fast", action="store_true")
        sp.set_defaults(func=cmd_validate, _command_name=name)

    sp = sub.add_parser(
        "forecast-sequential",
        help="Climatological Bayesian forecast (sequential, the production validation)",
    )
    sp.add_argument("--cities", nargs="+", default=None,
                    help="Cities to forecast (default: all available)")
    sp.add_argument("--cycles", nargs="+", type=int,
                    default=[2020, 2021, 2022, 2023, 2024],
                    help="Planting years to forecast (default: 2020..2024)")
    sp.add_argument("--profile", default="matopiba")
    sp.add_argument("--method", choices=["spei", "tercile"], default="spei",
                    help="Season classification method (default: spei)")
    sp.add_argument("--n-sim", type=int, default=500,
                    help="Monte-Carlo simulations per cycle (default: 500)")
    sp.add_argument("--alpha-init-from", default=None,
                    help="Path to JSON list with initial Dirichlet alpha "
                         "(default: uniform [1, 1, 1])")
    sp.add_argument("--output-dir", default="output/forecast_sequential")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_forecast_sequential)

    sp = sub.add_parser("sensitivity", help="Run Sobol' sensitivity analysis")
    sp.add_argument("--city", default="Balsas")
    sp.add_argument("--cycle", type=int, default=2023)
    sp.add_argument("--profile", default="matopiba")
    sp.add_argument("--n", type=int, default=1000)
    sp.add_argument("--output", default="output/sensitivity")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_sensitivity)

    sp = sub.add_parser("forecast", help="Fit + ensemble forecast for one cycle")
    sp.add_argument("--city", default="Balsas")
    sp.add_argument("--cycle", type=int, default=2023)
    sp.add_argument("--profile", default="matopiba")
    sp.add_argument("--draws", type=int, default=500)
    sp.add_argument("--tune", type=int, default=500)
    sp.add_argument("--chains", type=int, default=2)
    sp.add_argument("--members", type=int, default=31)
    sp.add_argument("--posterior-draws", type=int, default=200)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--output-dir", default="output/forecast")
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("figures", help="Regenerate publication figures")
    sp.add_argument("--results", default="output")
    sp.add_argument("--output", default="figures/paper")
    sp.add_argument("--no-pdf", action="store_true")
    sp.set_defaults(func=cmd_figures)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
