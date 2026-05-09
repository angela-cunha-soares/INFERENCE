"""Re-verify numerical claims in the manuscript against the source CSVs/JSONs."""
from __future__ import annotations

import csv
import json
import statistics

print("=== Posterior-recovery test (claimed: KGE 0.918, Cov 0.922, alpha-PIT 0.894) ===")
gm_rows = list(csv.DictReader(open("output/paper_tables/table_global_metrics.csv")))
gm = {r["metric"]: r for r in gm_rows}
print(f"  KGE median = {gm['KGE']['median']}, range {gm['KGE']['min']}--{gm['KGE']['max']}")
print(f"  Coverage_90 median = {gm[r'Coverage$_{90\%}$']['median']}")
print(f"  alpha-PIT mean = {gm[r'$\alpha$-PIT']['mean']}")
print(f"  CRPS median = {gm['CRPS']['median']}")

print("\n=== MCMC diagnostics (claimed: max R-hat <= 1.005, min ESS_bulk = 1028, 0 div) ===")
mcmc = list(csv.DictReader(open("output/paper_tables/table_mcmc_diagnostics.csv")))
rhats = [float(r["r_hat"]) for r in mcmc]
ess = [float(r["ess_bulk"]) for r in mcmc]
divs = [int(r["n_divergent"]) for r in mcmc]
print(f"  max R-hat = {max(rhats):.4f}")
print(f"  min ESS_bulk = {min(ess):.0f}")
print(f"  total divergent = {sum(divs)}")
draws_per_param = sum(int(r["n_chain"]) * int(r["n_draw"]) for r in mcmc if r["parameter"] == "theta_s")
print(f"  total draws per parameter across 5 cycles = {draws_per_param}")

print("\n=== Sequential forecast (claimed: CRPS median 18.2 mm, CRPSS +0.034, cov SW 0.954) ===")
seq = list(csv.DictReader(open("output/forecast_sequential/sequential_summary_all_cities.csv")))


def col(name: str, fn=statistics.mean) -> float:
    return fn([float(r[name]) for r in seq])


print(f"  n combinations = {len(seq)}")
print(f"  CRPS_I_total mean = {col('prob_CRPS_I_total_mm'):.3f}, median = {col('prob_CRPS_I_total_mm', statistics.median):.3f}")
print(f"  CRPSS vs naive climatology: mean = {col('CRPSS_vs_naive_climatology'):.4f}")
print(f"  Coverage SW daily: mean = {col('prob_coverage_90_SW_daily'):.4f}")
print(f"  Coverage I_total: mean = {col('prob_coverage_90_I_total'):.3f}")
print(f"  PIT I_total: mean = {col('prob_PIT_I_total'):.3f}, median = {col('prob_PIT_I_total', statistics.median):.3f}")
print(f"  KGE SW daily: mean = {col('det_KGE_SW'):.3f}")
print(f"  I_total signed error: mean = {col('det_I_total_error_mm'):.3f}, median |err| = {col('det_I_total_error_mm', lambda v: statistics.median([abs(x) for x in v])):.3f}")

print("\n=== Sobol' (claimed: theta_s S1=0.734, theta_init=0.459) ===")
s = json.load(open("output/sensitivity/sobol_Balsas_2023.json"))
print(f"  S1 theta_s    = {s['first_order']['theta_s']:.4f}")
print(f"  S1 theta_init = {s['first_order']['theta_init']:.4f}")
print(f"  S1 Kc_mult    = {s['first_order']['Kc_mult']:.4f}")
print(f"  S1 theta_r    = {s['first_order']['theta_r']:.4f}")

print("\n=== Backtest rolling (claimed: KGE +0.32, CRPS 2.65, cov 0.970) ===")
br = list(csv.DictReader(open("output/backtest_rolling/backtest_rolling_h5d.csv")))
crps = [float(r["crps_I_total"]) for r in br]
in90 = [r["in_90ci"] == "True" for r in br]
print(f"  total forecasts = {len(br)}")
print(f"  CRPS mean       = {statistics.mean(crps):.3f}")
print(f"  CRPS median     = {statistics.median(crps):.3f}")
print(f"  Coverage_90     = {sum(in90) / len(in90):.4f}")

# Per-cycle CRPS
print("\n=== Per-cycle CRPS (manuscript: 2021/22=24.7, 2022/23 and 2023/24 ~17) ===")
by_cycle: dict[str, list[float]] = {}
for r in seq:
    by_cycle.setdefault(r["cycle"], []).append(float(r["prob_CRPS_I_total_mm"]))
for k in sorted(by_cycle):
    print(f"  {k}: mean CRPS = {statistics.mean(by_cycle[k]):.2f} mm")

# Per-city CRPS
print("\n=== Per-city CRPS (manuscript: Tasso 11.7, LEM 33.3) ===")
by_city: dict[str, list[float]] = {}
for r in seq:
    by_city.setdefault(r["city"], []).append(float(r["prob_CRPS_I_total_mm"]))
for k in sorted(by_city):
    print(f"  {k}: mean CRPS = {statistics.mean(by_city[k]):.2f} mm")
