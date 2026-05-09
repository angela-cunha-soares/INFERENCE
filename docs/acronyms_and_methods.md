# Acronyms and Methods — paper-ready glossary

This document is the canonical reference for all acronyms, parameters,
equations and modelling choices used in the manuscript. It is intended to be
incorporated as the *Materials and Methods* section (or its appendix) of the
paper. Every symbol used in the equations of [methodology.md](methodology.md)
appears here with a one-line definition and a citation to the primary source.

---

## 1. Acronym table

| Acronym | Full name | Domain | Units / Range | Reference |
|---|---|---|---|---|
| **AWC** | Available Water Capacity | Soil hydrology | mm of plant-available water in the root zone | Allen et al. (1998), FAO-56 |
| **CAD** | Capacidade de Água Disponível | Soil hydrology (Portuguese) | m³ m⁻³ (volumetric) | Santos et al. (TED ANA-UFPR) |
| **ETo** | Reference evapotranspiration | Crop water demand | mm day⁻¹ | Allen et al. (1998), FAO-56 (Penman-Monteith) |
| **ETc** | Crop evapotranspiration (= Kc × ETo) | Crop water demand | mm day⁻¹ | FAO-56 Eq. 56 |
| **Kc** | Crop coefficient | Crop water demand | dimensionless, ∈ [0.3, 1.3] | FAO-56 Chapter 6 (Allen et al. 1998) |
| **MAD** | Management Allowed Depletion | Irrigation scheduling | fraction of AWC, ∈ [0, 1]; 0.55 for soybean | FAO-56 Table 22 |
| **Z_r** | Effective root depth | Crop physiology | cm; 60 cm for 90-day soybean | FAO-56 Table 22 |
| **θ_FC** | Volumetric water content at field capacity | Soil hydrology | m³ m⁻³ | Saxton & Rawls (2006) |
| **θ_PMP** | Volumetric water content at permanent wilting point | Soil hydrology | m³ m⁻³ | Saxton & Rawls (2006) |
| **θ_s, θ_r** | Saturated / residual volumetric water content | van Genuchten retention | m³ m⁻³ | van Genuchten (1980) |
| **α, m, n** | van Genuchten retention fitting parameters | Soil hydraulics | (kPa⁻¹, —, —) | van Genuchten (1980) |
| **SPEI** | Standardized Precipitation-Evapotranspiration Index | Climate / drought | dimensionless N(0,1) | Vicente-Serrano et al. (2010) |
| **D** | Climatic water balance, D = P − ETo | Climate | mm | Vicente-Serrano et al. (2010) |
| **P_eff** | Effective precipitation (factor × P) | Soil water budget | mm; factor 0.8 (FAO simplified) | FAO-56 Eq. 84 |
| **CRPS** | Continuous Ranked Probability Score | Forecast verification | mm (lower is better) | Hersbach (2000) |
| **CRPSS** | CRPS Skill Score, 1 − CRPS_model / CRPS_baseline | Forecast verification | dimensionless; >0 = beats baseline | Wilks (2011) |
| **PIT** | Probability Integral Transform | Forecast calibration | ∈ [0, 1]; uniform if calibrated | Gneiting et al. (2007) |
| **KGE** | Kling-Gupta Efficiency | Hydrologic verification | dimensionless; 1 = perfect | Kling & Gupta (2009) |
| **NSE** | Nash-Sutcliffe Efficiency | Hydrologic verification | dimensionless; 1 = perfect | Nash & Sutcliffe (1970) |
| **ENSO** | El Niño-Southern Oscillation | Climate, Pacific | — | Trenberth (1997) |
| **ONI** | Oceanic Niño Index, 3-mo running mean of Niño 3.4 SSTA | Pacific SST | °C, ∈ [−2.5, +2.5] typ. | NOAA CPC |
| **MEI** | Multivariate ENSO Index v2 | Pacific atmosphere-ocean | dimensionless | NOAA PSL (Wolter & Timlin 2011) |
| **SOI** | Southern Oscillation Index, Tahiti − Darwin SLP | Pacific atmospheric | dimensionless, ∈ [−4, +4] | NOAA CPC / Troup BoM |
| **TNA** | Tropical North Atlantic Index | Atlantic SST | °C anomaly | NOAA PSL |
| **TSA** | Tropical South Atlantic Index | Atlantic SST | °C anomaly | NOAA PSL |
| **AMM** | Atlantic Meridional Mode (SST projection) | Atlantic ocean-atmosphere | dimensionless | Chiang & Vimont (2004) |
| **AMO** | Atlantic Multidecadal Oscillation | Atlantic SST (low-frequency) | °C anomaly | Enfield et al. (2001) |
| **NAO** | North Atlantic Oscillation | Extratropical Atlantic | dimensionless, ∈ [−4, +4] typ. | Hurrell (1995); CPC |
| **ITCZ** | Intertropical Convergence Zone | Atmospheric circulation | latitude position | Hastenrath (1991) |
| **MATOPIBA** | MA-TO-PI-BA agricultural frontier | Brazilian context | states of MA, TO, PI, BA | Buainain & Garcia (2015) |
| **HYBRAS** | Hydrophysical database for Brazilian soils v1.0 | Soils, Brazil | 3,777 profiles | Ottoni et al. (2018), CPRM |
| **RADAM** | Brazilian soil cartographic database, 1:250,000 | Soils, Brazil | polygon shapefile | IBGE / BDIA |
| **SNIRH** | Sistema Nacional de Informações sobre Recursos Hídricos | Hydrology, Brazil | online catalog | ANA |
| **PTF** | Pedotransfer Function | Soil hydraulics | function (texture → hydraulics) | Saxton et al. (1986); Saxton & Rawls (2006) |
| **MCMC** | Markov Chain Monte Carlo | Bayesian inference | n_chains × n_draws | Brooks et al. (2011) |
| **NUTS** | No-U-Turn Sampler (HMC variant) | Bayesian inference | — | Hoffman & Gelman (2014) |
| **PyMC** | Probabilistic programming framework v5 | Software | Python package | Salvatier et al. (2016) |
| **Dir**, **Mult** | Dirichlet, Multinomial distributions | Probability | — | Gelman et al. (2013) |
| **R̂** (Rhat) | Potential scale reduction factor | MCMC convergence | ≈1 when converged | Gelman & Rubin (1992) |
| **ESS** | Effective Sample Size | MCMC convergence | usually >400 acceptable | Gelman et al. (2013) |

---

## 2. The water balance equation (FAO-56 simplified)

Daily soil water content **SW** (mm) in the root zone evolves as:

$$
\text{SW}_d = \min\!\Bigl(\text{AWC},\; \max\!\bigl(0,\; \text{SW}_{d-1} + P_{\text{eff},d} - \text{ETc}_d\bigr)\Bigr)
$$

with

- $P_{\text{eff},d} = 0.8 \cdot P_d$ (FAO-56 simplified)
- $\text{ETc}_d = K_{c,d} \cdot \text{ETo}_d$
- Irrigation event $I_d$ triggered when $\text{SW}_d < \text{AWC}\,(1 - \text{MAD})$, refilling to AWC.
- Deep percolation $\text{DP}_d = \max(0,\; \text{SW}_{d-1} + P_{\text{eff},d} - \text{ETc}_d - \text{AWC})$.

Implementation: [src/bwb/models/fao56.py](../src/bwb/models/fao56.py).

---

## 3. The crop coefficient (Kc) curve

We use the **simplified time-averaged 5-stage step formulation** of FAO-56
Chapter 6 (Allen et al. 1998), as adopted by Brazilian soybean agronomy
(Embrapa Soja convention; Steduto et al. 2012, FAO-66). Each phenological
phase has a constant $K_c$ value:

| Phase | Days | $K_c$ |
|---|---|---|
| Initial | 15 | 0.40 |
| Development | 15 | 0.80 |
| Mid-season | 40 | 1.15 |
| Late-season | 15 | 0.80 |
| Harvest | 5 | 0.50 |

Total cycle: 90 days; $\bar{K_c} \approx 0.87$. Planting on December 1.

Implementation: [src/bwb/phenology/kc_curves.py:fao56_kc_5stage_step](../src/bwb/phenology/kc_curves.py).

---

## 4. Soil hydraulics (van Genuchten 1980)

The retention curve relating volumetric water content θ to matric potential h
is:

$$
\theta(h) = \theta_r + (\theta_s - \theta_r)\,\bigl[1 + (\alpha h)^n\bigr]^{-m}
$$

We use the regional reference profile of *Latossolo Vermelho, sandy phase,
Sertãozinho series* (representative of MATOPIBA latosols) at three depths:
0–20 cm, 20–40 cm, 40–60 cm. Numerical values in
[matopiba.toml `[soil.van_genuchten.*]`](../src/bwb/config/regional/matopiba.toml).

Field capacity $\theta_{FC}$ corresponds to pF = 2.0 ($h$ ≈ −10 kPa);
permanent wilting point $\theta_{PMP}$ to pF = 4.2 ($h$ ≈ −1500 kPa).

---

## 5. Per-municipality available water capacity (AWC)

We use the national CAD dataset of **ANA-UFPR** (Santos et al., TED), in which
volumetric AWC for every Brazilian municipality is computed as
$\text{AWC} = \theta_{FC} - \theta_{PMP}$ via the Saxton & Rawls (2006)
pedotransfer function applied to HYBRAS (Ottoni et al. 2018) soil profiles
and extrapolated through the 1:250,000 RADAM/IBGE soil map. The reported
volumetric value is integrated over the 0–120 cm reference root zone.

For the 90-day early-cycle soybean, we use $Z_r = 60$ cm and:

$$
\text{AWC}_{\text{mm}}(c) = \text{CAD}_c\,(\text{m}^3/\text{m}^3) \times Z_r\,(\text{cm}) \times 10
$$

where $c$ indexes the city. Per-city values for the 10 MATOPIBA
municipalities are tabulated in
[output/paper_tables/table_soils_matopiba.tex](../output/paper_tables/table_soils_matopiba.tex)
(range 34–61 mm, mean 47 mm; sourced from
[data/soils/matopiba_cad.csv](../data/soils/matopiba_cad.csv)).

---

## 6. Standardized Precipitation-Evapotranspiration Index (SPEI)

For each historical crop cycle (1961–year of forecast minus 1) we compute:

1. **Cycle water balance**: $D_y = P_y^{\text{tot}} - \text{ETo}_y^{\text{tot}}$ (mm).
2. **Distribution fit**: log-logistic (Fisk) on shifted $D_y$ values.
3. **SPEI**: $\text{SPEI}_y = \Phi^{-1}(F_{\text{LL}}(D_y))$ where $\Phi$ is the standard
   Normal CDF (Vicente-Serrano et al. 2010).

The SPEI series is then partitioned into terciles to assign each season a
class $\kappa \in \{0=\text{dry}, 1=\text{normal}, 2=\text{wet}\}$.

Implementation: [src/bwb/forecast/climatological.py:compute_seasonal_spei](../src/bwb/forecast/climatological.py).

---

## 7. The Bayesian Dirichlet-Multinomial model

### 7.1 Likelihood

Let $n_k(t)$ = count of historical seasons in $\{1961, \dots, t-1\}$ classified
as class $k$. Then the conjugate Bayesian update for the seasonal class
weights $w = (w_0, w_1, w_2)$ is:

$$
\begin{aligned}
w &\sim \text{Dir}(\alpha) \\
n &\sim \text{Mult}(N, w) \\
w \mid n &\sim \text{Dir}(\alpha + n)
\end{aligned}
$$

### 7.2 Climatologically informed prior (default since v0.2)

$$
\alpha_0 = \mathbf{1} + \bigl(n_0(1961, t_0-1),\; n_1(1961, t_0-1),\; n_2(1961, t_0-1)\bigr)
$$

where $t_0 = \min(\text{target years})$ (e.g. 2020) and the additive 1 is a
Laplace-type smoother that prevents zero counts. For our test
period (2020–2024 forecasts), $\alpha_0 \approx (21, 20, 21)$ — the prior
carries 59 years of observed seasonal classes from MATOPIBA Xavier
reanalysis.

Implementation: [src/bwb/forecast/climatological.py:run_sequential_forecast](../src/bwb/forecast/climatological.py).

### 7.3 Sequential update

For each target cycle $c$ in $\{2020, 2021, ..., 2024\}$:

1. Sample $w \sim \text{Dir}(\alpha_c)$ for $N_{\text{sim}}$ Monte-Carlo draws.
2. For each draw, sample class $\kappa \sim \text{Cat}(w)$ then resample one
   complete historical season $(P_d, \text{ETo}_d)$ of class $\kappa$.
3. Run the FAO-56 daily water balance with that season's forcings.
4. Aggregate the $N_{\text{sim}}$ trajectories → distribution of $\text{SW}_d$
   and $I_d$.
5. After observing the realised cycle's class $\kappa^*$, update
   $\alpha_{c+1} = \alpha_c + e_{\kappa^*}$.

### 7.4 What this model is — and is not

**It is** a Bayesian climatological estimator of the seasonal class
prevalence, updated annually. The forecast at each $c$ is a *Monte-Carlo
ensemble* of FAO-56 trajectories weighted by the posterior $w$. CRPS on
total irrigation captures the operationally meaningful skill: how well the
ensemble brackets the true irrigation requirement.

**It is not** a year-by-year predictor of the SPEI tercile. Without
predictive covariates (ENSO/Atlantic SST, antecedent soil moisture, etc.),
the climatological prior dominates and the predicted class probabilities
remain near the long-term frequency $(1/3, 1/3, 1/3)$. This is why
classification accuracy and Cohen's $\kappa$ are near zero — by
construction, not by failure. We discuss this limitation in
[Section 9](#9-limitations-and-future-work).

---

## 8. Forecast verification

### 8.1 Continuous Ranked Probability Score (CRPS)

For a forecast CDF $F$ and verifying observation $y$:

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \bigl(F(x) - \mathbb{1}\{x \ge y\}\bigr)^2\,dx
$$

For an ensemble forecast we use the empirical estimator (Hersbach 2000).
Lower is better; units of CRPS match the forecast variable (mm of
irrigation in our case).

### 8.2 CRPS Skill Score (CRPSS)

$$
\text{CRPSS} = 1 - \frac{\text{CRPS}_{\text{model}}}{\text{CRPS}_{\text{baseline}}}
$$

where the baselines are: (a) **naive climatology** (uniform sample over
1961–2019 historical cycles); (b) **persistence** (last cycle's value);
(c) **long-term mean** (point forecast at the 1961–2019 mean).
$\text{CRPSS} > 0$ means the model beats the baseline.

### 8.3 Probability Integral Transform (PIT) and reliability

For each forecast we compute $\text{PIT} = F_{\text{forecast}}(y)$. A perfectly
calibrated forecast yields PIT values uniformly distributed on $[0, 1]$.
Deviations (U-shape, J-shape, hump) diagnose under/over-confidence.

### 8.4 Coverage and interval width

The fraction of observations falling inside the forecast 90% prediction
interval; the nominal level is 0.90. Soil-moisture coverage is assessed
day-by-day; total-irrigation coverage is per-cycle.

---

## 9. Limitations and future work

1. **No predictive covariates.** The Dirichlet-Multinomial prior is
   informed only by climatology of past seasons, not by predictive
   covariates available *before* the cycle starts (ENSO state at
   November, Atlantic SST gradient, antecedent November precipitation,
   etc.). Adding these as covariates of a multinomial logit on $w_c$ is
   the highest-impact extension.
2. **Soil heterogeneity within each city** is not represented; we use the
   municipality-mean CAD from ANA. Within-city variability (different soil
   classes by relief or land use) is averaged out.
3. **Per-city van Genuchten parameters** would require extracting
   municipality-specific HYBRAS profiles, currently approximated by the
   regional Sertãozinho reference for the entire MATOPIBA.
4. **Operational forecasting** (2025/26 cycle and beyond) requires the
   `--alpha-init-from output/forecast_sequential/alpha_final_<city>.json`
   workflow already implemented.

---

## 10. References

Allen, R.G.; Pereira, L.S.; Raes, D.; Smith, M. (1998). *Crop
evapotranspiration — Guidelines for computing crop water requirements*.
FAO Irrigation and Drainage Paper 56. Rome: FAO.

Chiang, J.C.H.; Vimont, D.J. (2004). Analogous Pacific and Atlantic
meridional modes of tropical atmosphere-ocean variability. *J. Climate*
17(21), 4143–4158.

Enfield, D.B.; Mestas-Nuñez, A.M.; Trimble, P.J. (2001). The Atlantic
multidecadal oscillation and its relation to rainfall and river flows in
the continental U.S. *Geophys. Res. Lett.* 28(10), 2077–2080.

Gneiting, T.; Balabdaoui, F.; Raftery, A.E. (2007). Probabilistic
forecasts, calibration and sharpness. *J. R. Stat. Soc. B* 69(2), 243–268.

Hastenrath, S. (1991). *Climate Dynamics of the Tropics*. Kluwer.

Hersbach, H. (2000). Decomposition of the continuous ranked probability
score for ensemble prediction systems. *Wea. Forecasting* 15(5), 559–570.

Hurrell, J.W. (1995). Decadal trends in the North Atlantic Oscillation.
*Science* 269(5224), 676–679.

Kling, H.; Gupta, H. (2009). On the development of regionalization
relationships for lumped watershed models. *J. Hydrol.* 373(3-4), 337–351.

Nash, J.E.; Sutcliffe, J.V. (1970). River flow forecasting through
conceptual models part I — A discussion of principles. *J. Hydrol.* 10(3),
282–290.

Ottoni, M.V.; Ottoni-Filho, T.B.; Schaap, M.G.; Lopes-Assad, M.L.R.C.;
Rotunno-Filho, O.C. (2018). HYBRAS: hydrophysical database for Brazilian
soils v1.0. CPRM partial report.

Salvatier, J.; Wiecki, T.V.; Fonnesbeck, C. (2016). Probabilistic
programming in Python using PyMC3. *PeerJ Computer Science* 2, e55.

Santos, I.; Siefert, C.A.C.; Marangon, F.H.S.; Schultz, G.B.; Lima, C.R.;
Fontenelle, T.H.; Ferreira, D.A.C.; Gonçalves, M.V.C. *Capacidade de Água
Disponível (CAD/AWC) para Solos Brasileiros*. TED ANA-UFPR, SNIRH record
28fe4baa-66f3-4f6b-b0d2-890abf5910c4.

Saxton, K.E.; Rawls, W.J.; Romberger, J.S.; Papendick, R.I. (1986).
Estimating generalized soil-water characteristics from texture. *SSSAJ*
50(4), 1031–1036.

Saxton, K.E.; Rawls, W.J. (2006). Soil water characteristic estimates by
texture and organic matter for hydrologic solutions. *SSSAJ* 70(5),
1569–1578.

Steduto, P.; Hsiao, T.C.; Fereres, E.; Raes, D. (2012). *Crop yield
response to water*. FAO Irrigation and Drainage Paper 66. Rome: FAO.

van Genuchten, M.Th. (1980). A closed-form equation for predicting the
hydraulic conductivity of unsaturated soils. *SSSAJ* 44(5), 892–898.

Vicente-Serrano, S.M.; Beguería, S.; López-Moreno, J.I. (2010). A
multiscalar drought index sensitive to global warming: the standardized
precipitation evapotranspiration index. *J. Climate* 23(7), 1696–1718.

Wilks, D.S. (2011). *Statistical Methods in the Atmospheric Sciences*
(3rd ed.). Academic Press.

Wolter, K.; Timlin, M.S. (2011). El Niño/Southern Oscillation behaviour
since 1871 as diagnosed in an extended multivariate ENSO index (MEI.ext).
*Int. J. Climatol.* 31(7), 1074–1087.
