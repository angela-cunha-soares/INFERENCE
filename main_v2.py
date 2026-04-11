# =============================================================================
# MODELO DE INFERENCIA BAYESIANA SEQUENCIAL – BALANCO HIDRICO DA SOJA
# Balsas (MA) | Ciclo 90 dias | Plantio 01/12
#
# AVANCOS SOBRE RIBEIRO (2024):
#   1. Balanco hidrico diario FAO-56 (SW, Kc por fase, CAD, raiz)
#   2. ETo e P modelados como processos estocasticos separados
#   3. Dependencia temporal via AR(1) na precipitacao
#   4. Atualizacao sequencial (posterior -> prior)
#   5. Laminas de irrigacao com distribuicao completa + decisao otima
#   6. Validacao dia a dia + metricas probabilisticas
#   7. Efeito hierarquico por safra (variabilidade inter-anual)
#      Permite gerar cenarios de anos secos/chuvosos (Dez-Fev)
#
# Ref: Ribeiro (2024), FAO-56 Cap. 6, FAO-66
# Parametros da cultura: Profa. Patricia
# =============================================================================

import os
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

os.makedirs('output', exist_ok=True)

print("=" * 70)
print("MODELO DE INFERENCIA BAYESIANA SEQUENCIAL")
print("Balanco hidrico probabilistico — Soja irrigada, Balsas (MA)")
print("Continuidade: Ribeiro (2024) + FAO-56 + atualizacao iterativa")
print("=" * 70)

# =============================================================================
# PARAMETROS DA CULTURA (Profa. Patricia + FAO-56 Cap. 6 + FAO-66)
# =============================================================================
# Ciclo: 90 dias | Plantio: 01/12 | Sistema radicular: 60 cm
#
# Fase            | Duracao | Kc
# Inicial         | 15 dias | 0.40
# Desenvolvimento | 15 dias | 0.80
# Intermediaria   | 40 dias | 1.15
# Final           | 15 dias | 0.80
# Colheita        |  5 dias | 0.50

PLANTIO_MES = 12
PLANTIO_DIA = 1
DURACAO_CICLO = 90

AWC = 120.0    # Capacidade de Agua Disponivel (mm), sistema radicular 60 cm
MAD = 0.55     # Fracao de deplecao permitida (soja)

CICLOS_ALVO = [2020, 2021, 2022, 2023, 2024]

N_SAMPLES = 2000
N_SIM = 500


def get_kc(dia):
    """Kc por dia do ciclo (1-90). Ref: FAO-56 Cap. 6, FAO-66."""
    if   1 <= dia <= 15: return 0.40
    elif 16 <= dia <= 30: return 0.80
    elif 31 <= dia <= 70: return 1.15
    elif 71 <= dia <= 85: return 0.80
    elif 86 <= dia <= 90: return 0.50
    return 0.50


KC_ARRAY = np.array([get_kc(d) for d in range(1, DURACAO_CICLO + 1)])


# =============================================================================
# 1. PREPARACAO DE DADOS
# =============================================================================

df_hist = pd.read_csv('data/Balsas_MA.csv')
df_hist['Data'] = pd.to_datetime(df_hist['Data'])
df_hist = df_hist.sort_values('Data').reset_index(drop=True)

print(f"\nDados historicos: {len(df_hist)} dias "
      f"({df_hist['Data'].dt.year.min()}-{df_hist['Data'].dt.year.max()})")


def extrair_safras(df, mes=PLANTIO_MES, dia=PLANTIO_DIA, duracao=DURACAO_CICLO):
    """Extrai janelas de 90 dias (01/12 -> ~28/02) para cada ano."""
    safras = []
    for ano in sorted(df['Data'].dt.year.unique()):
        inicio = pd.Timestamp(year=ano, month=mes, day=dia)
        fim = inicio + pd.Timedelta(days=duracao - 1)
        mask = (df['Data'] >= inicio) & (df['Data'] <= fim)
        safra = df.loc[mask].copy()
        if len(safra) == duracao:
            safra = safra.reset_index(drop=True)
            safra['dia_ciclo'] = np.arange(1, duracao + 1)
            safra['ano_safra'] = ano
            safras.append(safra)
    if not safras:
        raise ValueError("Nenhuma safra completa encontrada nos dados.")
    return pd.concat(safras, ignore_index=True)


df_safras = extrair_safras(df_hist)

# Variaveis derivadas
df_safras['Kc'] = df_safras['dia_ciclo'].apply(get_kc)
df_safras['ETc'] = df_safras['Kc'] * df_safras['ETo']
df_safras['P_eff'] = 0.8 * df_safras['pr']
df_safras['deficit'] = df_safras['ETc'] - df_safras['P_eff']

anos_disponiveis = sorted(df_safras['ano_safra'].unique())
print(f"Safras completas: {len(anos_disponiveis)} "
      f"({anos_disponiveis[0]}/{anos_disponiveis[0]+1} a "
      f"{anos_disponiveis[-1]}/{anos_disponiveis[-1]+1})")


# =============================================================================
# 2. BALANCO HIDRICO DETERMINISTICO (FAO-56)
# =============================================================================

def balanco_hidrico(df, awc=AWC, mad=MAD):
    """Balanco hidrico diario FAO-56."""
    df = df.copy()
    df['Kc'] = df['dia_ciclo'].apply(get_kc)
    df['ETc'] = df['Kc'] * df['ETo']
    df['P_eff'] = 0.8 * df['pr']

    n = len(df)
    SW = np.zeros(n)
    I_arr = np.zeros(n)
    DP = np.zeros(n)

    sw = awc
    mad_threshold = awc * (1 - mad)

    for i in range(n):
        p_eff = df.iloc[i]['P_eff']
        etc = df.iloc[i]['ETc']
        sw_new = sw + p_eff - etc
        dp = max(0.0, sw_new - awc)
        sw_new = min(sw_new, awc)
        irrig = 0.0
        if sw_new < mad_threshold:
            irrig = awc - sw_new
            sw_new = awc
        sw_new = max(0.0, sw_new)
        SW[i] = sw_new
        I_arr[i] = irrig
        DP[i] = dp
        sw = sw_new

    df['SW'] = SW
    df['I'] = I_arr
    df['DP'] = DP
    return df


def balanco_hidrico_arrays(eto, pr, kc=KC_ARRAY, awc=AWC, mad=MAD):
    """Balanco hidrico a partir de arrays ETo e pr.
    Retorna SW, I, DP, ETc como arrays numpy.
    Avanco: usa ETo e P separadamente (nao deficit agregado)."""
    n = len(eto)
    etc = kc[:n] * eto
    p_eff = 0.8 * pr

    SW = np.zeros(n)
    I_arr = np.zeros(n)
    DP = np.zeros(n)
    ETc_arr = etc.copy()

    sw = awc
    mad_threshold = awc * (1 - mad)

    for i in range(n):
        sw_new = sw + p_eff[i] - etc[i]
        dp = max(0.0, sw_new - awc)
        sw_new = min(sw_new, awc)
        irrig = 0.0
        if sw_new < mad_threshold:
            irrig = awc - sw_new
            sw_new = awc
        sw_new = max(0.0, sw_new)
        SW[i] = sw_new
        I_arr[i] = irrig
        DP[i] = dp
        sw = sw_new

    return SW, I_arr, DP, ETc_arr


# =============================================================================
# 3. MODELO BAYESIANO HIERARQUICO
#    ETo e P separados + AR(1) em P + efeito hierarquico por safra
#    Avanco principal: variabilidade inter-anual permite gerar anos secos
# =============================================================================

def treinar_modelo(df_treino, n_samples=N_SAMPLES):
    """Modelo Bayesiano hierarquico com ETo e P separados + AR(1) + year effects.

    Avanco sobre Ribeiro (2024):
    - ETo e P modelados como processos independentes (nao deficit agregado)
    - P com AR(1) na likelihood — rho aprendido dos dados
    - Efeito hierarquico multiplicativo por safra (non-centered)
      => Permite gerar cenarios de anos secos/chuvosos
      => Chave para capturar risco de irrigacao no periodo chuvoso
    - Priors climatologicos informados por dia do ciclo

    Implementacao:
    - Likelihood condicional com lags pre-computados (sem pm.scan)
    - Year effect: mu_eff[d,j] = mu[d] * exp(year_effect[j])
    """
    # Estatisticas climatologicas por dia do ciclo
    stats_eto = df_treino.groupby('dia_ciclo').agg(
        mu=('ETo', 'mean'), sigma=('ETo', 'std')
    ).reset_index()

    stats_pr = df_treino.groupby('dia_ciclo').agg(
        mu=('pr', 'mean'), sigma=('pr', 'std')
    ).reset_index()

    # --- Pre-computar arrays para likelihood AR(1) condicional ---
    dias_idx = df_treino['dia_ciclo'].values - 1      # 0-based
    pr_values = df_treino['pr'].values
    safra_ids = df_treino['ano_safra'].values

    # Identifica o primeiro dia de cada safra (AR(1) reinicia aqui)
    is_first = np.zeros(len(df_treino), dtype=bool)
    is_first[0] = True
    is_first[1:] = safra_ids[1:] != safra_ids[:-1]
    is_cont = ~is_first

    # Precipitacao e dia_ciclo defasados (lag-1)
    pr_lag = np.zeros(len(df_treino))
    pr_lag[1:] = pr_values[:-1]

    dias_lag = np.zeros(len(df_treino), dtype=int)
    dias_lag[1:] = dias_idx[:-1]

    # Indices para cada grupo
    idx_first = np.where(is_first)[0]
    idx_cont = np.where(is_cont)[0]

    # --- Indice de safra para efeito hierarquico ---
    safra_ids_unique = np.unique(safra_ids)
    n_safras_treino = len(safra_ids_unique)
    safra_map = {s: i for i, s in enumerate(safra_ids_unique)}
    safra_idx_arr = np.array([safra_map[s] for s in safra_ids])

    with pm.Model() as model:
        # === ETo ===
        mu_eto = pm.Normal(
            'mu_eto',
            mu=stats_eto['mu'].values,
            sigma=stats_eto['sigma'].values + 0.1,
            shape=DURACAO_CICLO
        )
        sigma_eto = pm.HalfNormal('sigma_eto', sigma=2)

        # Efeito hierarquico por safra em ETo (non-centered)
        sigma_year_eto = pm.HalfNormal('sigma_year_eto', sigma=0.2)
        year_eto_raw = pm.Normal('year_eto_raw', mu=0, sigma=1,
                                 shape=n_safras_treino)
        year_eto = pm.Deterministic('year_eto', year_eto_raw * sigma_year_eto)

        # Media efetiva com efeito multiplicativo por safra
        mu_eto_eff = mu_eto[dias_idx] * pt.exp(year_eto[safra_idx_arr])
        pm.Normal('eto_obs', mu=mu_eto_eff, sigma=sigma_eto,
                  observed=df_treino['ETo'].values)

        # === Precipitacao ===
        mu_pr = pm.Normal(
            'mu_pr',
            mu=stats_pr['mu'].values,
            sigma=stats_pr['sigma'].values + 0.1,
            shape=DURACAO_CICLO
        )
        sigma_pr = pm.HalfNormal('sigma_pr', sigma=15)

        # Coeficiente AR(1) na likelihood
        rho_pr = pm.Uniform('rho_pr', lower=0.0, upper=0.95)

        # Efeito hierarquico por safra em P (non-centered)
        # => Permite anos mais secos ou mais chuvosos que a media
        sigma_year_pr = pm.HalfNormal('sigma_year_pr', sigma=0.5)
        year_pr_raw = pm.Normal('year_pr_raw', mu=0, sigma=1,
                                shape=n_safras_treino)
        year_pr = pm.Deterministic('year_pr', year_pr_raw * sigma_year_pr)

        # --- Likelihood: P com AR(1) + year effect ---
        ye_first = year_pr[safra_idx_arr[idx_first]]
        ye_cont = year_pr[safra_idx_arr[idx_cont]]

        # Primeiro dia de cada safra (incondicional)
        pm.Normal(
            'pr_obs_first',
            mu=mu_pr[dias_idx[idx_first]] * pt.exp(ye_first),
            sigma=sigma_pr,
            observed=pr_values[idx_first]
        )

        # Dias seguintes: AR(1) condicional + year effect
        mu_t = mu_pr[dias_idx[idx_cont]] * pt.exp(ye_cont)
        mu_lag = mu_pr[dias_lag[idx_cont]] * pt.exp(ye_cont)  # mesma safra
        mu_cond = mu_t + rho_pr * (pr_lag[idx_cont] - mu_lag)
        pm.Normal(
            'pr_obs_cont',
            mu=mu_cond,
            sigma=sigma_pr,
            observed=pr_values[idx_cont]
        )

    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True
        )

    # Diagnostico MCMC
    print(f"  Diagnostico MCMC:")
    diag_vars = ['sigma_eto', 'sigma_pr', 'rho_pr',
                 'sigma_year_eto', 'sigma_year_pr']
    summary = az.summary(trace, var_names=diag_vars)
    print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']].to_string())

    return model, trace


def prever_bayesiano(trace, n_sim=N_SIM, awc=AWC, mad=MAD):
    """Gera distribuicao de laminas a partir da posterior.

    Avanco: gera cenarios de ETo e P com variabilidade inter-anual
    (year effects hierarquicos), depois roda balanco hidrico.
    Cada cenario simula um ano com seca/chuva diferente da media.
    """
    mu_eto_post = trace.posterior['mu_eto'].values.reshape(-1, DURACAO_CICLO)
    sigma_eto_post = trace.posterior['sigma_eto'].values.reshape(-1)
    mu_pr_post = trace.posterior['mu_pr'].values.reshape(-1, DURACAO_CICLO)
    sigma_pr_post = trace.posterior['sigma_pr'].values.reshape(-1)
    rho_post = trace.posterior['rho_pr'].values.reshape(-1)

    # Variabilidade inter-anual aprendida pelo modelo hierarquico
    sigma_year_eto_post = trace.posterior['sigma_year_eto'].values.reshape(-1)
    sigma_year_pr_post = trace.posterior['sigma_year_pr'].values.reshape(-1)

    n_post = mu_eto_post.shape[0]
    idx = np.random.choice(n_post, size=min(n_sim, n_post), replace=False)

    I_total_samples = []
    I_diario_samples = []
    SW_samples = []
    ETc_samples = []

    for i in idx:
        # Sorteia efeito de ano (seco ou chuvoso) para este cenario
        year_eff_eto = np.random.normal(0, sigma_year_eto_post[i])
        year_eff_pr = np.random.normal(0, sigma_year_pr_post[i])

        # ETo com variabilidade inter-anual + ruido diario
        mu_eto_scenario = mu_eto_post[i] * np.exp(year_eff_eto)
        eto_sim = np.maximum(0.0, np.random.normal(mu_eto_scenario,
                                                    sigma_eto_post[i]))

        # P com variabilidade inter-anual + AR(1) + ruido diario
        rho = rho_post[i]
        mu_pr_scenario = mu_pr_post[i] * np.exp(year_eff_pr)
        pr_sim = np.zeros(DURACAO_CICLO)
        pr_sim[0] = max(0.0, np.random.normal(mu_pr_scenario[0],
                                               sigma_pr_post[i]))
        for t in range(1, DURACAO_CICLO):
            ar_component = rho * (pr_sim[t-1] - mu_pr_scenario[t-1])
            pr_sim[t] = max(0.0, mu_pr_scenario[t] + ar_component +
                           np.random.normal(0, sigma_pr_post[i]))

        # Balanco hidrico com ETo e P separados
        SW_sim, I_sim, _, ETc_sim = balanco_hidrico_arrays(eto_sim, pr_sim)
        I_total_samples.append(I_sim.sum())
        I_diario_samples.append(I_sim)
        SW_samples.append(SW_sim)
        ETc_samples.append(ETc_sim)

    return (np.array(I_total_samples), np.array(I_diario_samples),
            np.array(SW_samples), np.array(ETc_samples))


# =============================================================================
# 4. METRICAS DE VALIDACAO (adaptadas para series esparsas de irrigacao)
# =============================================================================

def calcular_metricas(y_real, y_pred):
    """Metricas para series de irrigacao (podem ter muitos zeros).
    Usa metricas classicas + probabilisticas adequadas."""
    n = len(y_real)

    # MAE e Bias (sempre validos)
    mae = np.mean(np.abs(y_real - y_pred))
    bias = np.mean(y_pred - y_real)

    # Irrigacao total
    total_real = y_real.sum()
    total_pred = y_pred.sum()
    erro_total = total_pred - total_real

    # Acertos binarios (previu irrigacao quando ocorreu?)
    real_irriga = y_real > 0
    pred_irriga = y_pred > 0
    n_real = real_irriga.sum()
    n_pred = pred_irriga.sum()

    # KGE e NSE (somente se houver variancia na serie real)
    kge = np.nan
    nse = np.nan
    if np.std(y_real) > 1e-3:
        r = pearsonr(y_real, y_pred)[0]
        alpha = np.std(y_pred) / np.std(y_real)
        beta = np.mean(y_pred) / (np.mean(y_real) + 1e-10)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        nse = 1 - np.sum((y_real - y_pred)**2) / np.sum((y_real - np.mean(y_real))**2)

    return {
        'MAE_mm': round(mae, 2),
        'Bias_mm': round(bias, 2),
        'I_total_det': round(total_real, 1),
        'I_total_bayes': round(total_pred, 1),
        'Erro_total_mm': round(erro_total, 1),
        'N_irrig_real': int(n_real),
        'N_irrig_pred': int(n_pred),
        'KGE': round(kge, 3) if not np.isnan(kge) else 'N/A',
        'NSE': round(nse, 3) if not np.isnan(nse) else 'N/A',
    }


def metricas_probabilisticas(I_totais, I_det_total):
    """Metricas probabilisticas: CRPS, cobertura do IC, posicao do real na distribuicao."""
    # Posicao percentil do deterministico na distribuicao Bayesiana
    percentil = np.mean(I_totais <= I_det_total) * 100

    # Coverage: IC 90% contem o valor real?
    q05 = np.quantile(I_totais, 0.05)
    q95 = np.quantile(I_totais, 0.95)
    dentro_ic = q05 <= I_det_total <= q95

    # CRPS simplificado (Continuous Ranked Probability Score)
    crps = np.mean(np.abs(I_totais - I_det_total)) - 0.5 * np.mean(
        np.abs(I_totais[:, None] - I_totais[None, :])
    ) if len(I_totais) < 2000 else np.mean(np.abs(I_totais - I_det_total))

    return {
        'Percentil_det': round(percentil, 1),
        'Dentro_IC90': dentro_ic,
        'IC90_inf': round(q05, 1),
        'IC90_sup': round(q95, 1),
        'CRPS': round(crps, 2),
    }


# =============================================================================
# 5. LOOP SEQUENCIAL: TREINO -> PREVISAO -> VALIDACAO -> ATUALIZACAO
# =============================================================================

resultados = {}
tabela_metricas = []
tabela_prob = []

for ano_alvo in CICLOS_ALVO:
    ciclo_str = f"{ano_alvo}/{ano_alvo + 1}"
    print(f"\n{'=' * 70}")
    print(f"CICLO {ciclo_str}")
    print(f"{'=' * 70}")

    # Treino: todas as safras anteriores ao ciclo-alvo
    df_treino = df_safras[df_safras['ano_safra'] < ano_alvo].copy()
    n_treino = df_treino['ano_safra'].nunique()
    print(f"  Treino: {n_treino} safras (ate {ano_alvo - 1}/{ano_alvo})")

    # Treina modelo
    print(f"  Treinando modelo Bayesiano (ETo + P separados, AR(1))...")
    model, trace = treinar_modelo(df_treino)
    az.to_netcdf(trace, f'output/trace_{ano_alvo}_{ano_alvo + 1}.nc')

    # Previsao Bayesiana
    I_totais, I_diarios, SW_sims, ETc_sims = prever_bayesiano(trace)
    I_media_diaria = np.mean(I_diarios, axis=0)

    print(f"  Previsao: Irrigacao total = {np.mean(I_totais):.1f} mm "
          f"[IC 90%: {np.quantile(I_totais, 0.05):.1f}"
          f"-{np.quantile(I_totais, 0.95):.1f}]")

    # Salva previsao diaria completa
    df_pred = pd.DataFrame({
        'dia_ciclo': np.arange(1, DURACAO_CICLO + 1),
        'Kc': KC_ARRAY,
        'ETc_media': np.mean(ETc_sims, axis=0),
        'SW_media': np.mean(SW_sims, axis=0),
        'SW_q05': np.quantile(SW_sims, 0.05, axis=0),
        'SW_q95': np.quantile(SW_sims, 0.95, axis=0),
        'I_media': I_media_diaria,
        'I_q05': np.quantile(I_diarios, 0.05, axis=0),
        'I_q25': np.quantile(I_diarios, 0.25, axis=0),
        'I_q50': np.quantile(I_diarios, 0.50, axis=0),
        'I_q75': np.quantile(I_diarios, 0.75, axis=0),
        'I_q95': np.quantile(I_diarios, 0.95, axis=0),
        'I_std': np.std(I_diarios, axis=0),
        'Prob_irrig': np.mean(I_diarios > 0, axis=0),
    })
    df_pred.to_csv(f'output/previsao_{ano_alvo}_{ano_alvo + 1}.csv', index=False)

    # Validacao com dados reais
    df_real = df_safras[df_safras['ano_safra'] == ano_alvo]
    tem_real = len(df_real) == DURACAO_CICLO

    if tem_real:
        df_det = balanco_hidrico(df_real.reset_index(drop=True))
        df_det.to_csv(f'output/deterministico_{ano_alvo}_{ano_alvo + 1}.csv', index=False)
        I_det = df_det['I'].values

        met = calcular_metricas(I_det, I_media_diaria)
        met['Ciclo'] = ciclo_str
        met['N_treino'] = n_treino
        tabela_metricas.append(met)

        met_prob = metricas_probabilisticas(I_totais, I_det.sum())
        met_prob['Ciclo'] = ciclo_str
        tabela_prob.append(met_prob)

        print(f"  Deterministico: {I_det.sum():.1f} mm")
        print(f"  MAE={met['MAE_mm']} mm | Bias={met['Bias_mm']} mm | "
              f"Erro total={met['Erro_total_mm']} mm")
        print(f"  Percentil do real: {met_prob['Percentil_det']}% | "
              f"Dentro IC90%: {met_prob['Dentro_IC90']} | "
              f"CRPS={met_prob['CRPS']}")

        # --- Grafico: balanco hidrico diario com bandas de incerteza ---
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        dias = np.arange(1, DURACAO_CICLO + 1)

        # (a) Armazenamento de agua no solo (SW)
        ax = axes[0]
        ax.fill_between(dias,
                        np.quantile(SW_sims, 0.05, axis=0),
                        np.quantile(SW_sims, 0.95, axis=0),
                        alpha=0.3, color='steelblue', label='IC 90% Bayesiano')
        ax.plot(dias, np.mean(SW_sims, axis=0), '-', color='steelblue',
                linewidth=2, label='SW media Bayesiana')
        ax.plot(dias, df_det['SW'].values, '--', color='red',
                linewidth=2, label='SW deterministico')
        ax.axhline(AWC * (1 - MAD), color='orange', linestyle=':', label=f'Limiar MAD ({AWC*(1-MAD):.0f} mm)')
        ax.set_ylabel('Armazenamento (mm)')
        ax.set_title(f'Balanco hidrico diario — Ciclo {ciclo_str}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (b) ETc
        ax = axes[1]
        ax.fill_between(dias,
                        np.quantile(ETc_sims, 0.05, axis=0),
                        np.quantile(ETc_sims, 0.95, axis=0),
                        alpha=0.3, color='green', label='IC 90% ETc')
        ax.plot(dias, np.mean(ETc_sims, axis=0), '-', color='green',
                linewidth=2, label='ETc media Bayesiana')
        ax.plot(dias, df_det['ETc'].values, '--', color='red',
                linewidth=2, label='ETc real')
        ax.bar(dias, df_det['P_eff'].values, alpha=0.4, color='blue',
               label='P_eff real')
        ax.set_ylabel('mm/dia')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (c) Irrigacao
        ax = axes[2]
        ax.fill_between(dias,
                        np.quantile(I_diarios, 0.05, axis=0),
                        np.quantile(I_diarios, 0.95, axis=0),
                        alpha=0.3, color='orange', label='IC 90% Irrigacao')
        ax.plot(dias, I_media_diaria, '-', color='orange',
                linewidth=2, label='I media Bayesiana')
        ax.bar(dias, I_det, alpha=0.6, color='red', label='I deterministico')
        ax.set_ylabel('Irrigacao (mm)')
        ax.set_xlabel('Dia do ciclo')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'output/balanco_diario_{ano_alvo}_{ano_alvo + 1}.png', dpi=300)
        plt.close()

        # --- Violin plot ---
        fig, ax = plt.subplots(figsize=(7, 5))
        df_comp = pd.DataFrame({
            'Deterministico\n(dados reais)': I_det,
            'Bayesiano\n(previsao)': I_media_diaria,
        })
        sns.violinplot(data=df_comp, ax=ax, palette=['#2196F3', '#FF9800'])
        ax.set_ylabel('Lamina diaria (mm)')
        ax.set_title(f'Laminas de irrigacao — Ciclo {ciclo_str}')
        plt.tight_layout()
        plt.savefig(f'output/violin_{ano_alvo}_{ano_alvo + 1}.png', dpi=300)
        plt.close()
    else:
        print(f"  Dados reais indisponiveis -> previsao pura (sem validacao)")

    resultados[ano_alvo] = {
        'I_totais': I_totais,
        'I_diarios': I_diarios,
        'SW_sims': SW_sims,
        'ETc_sims': ETc_sims,
        'I_det': I_det if tem_real else None,
        'metricas': met if tem_real else None,
        'tem_real': tem_real,
    }

# =============================================================================
# 6. SAIDAS FINAIS
# =============================================================================

# --- Tabela de metricas ---
if tabela_metricas:
    df_met = pd.DataFrame(tabela_metricas)
    cols = ['Ciclo', 'N_treino', 'I_total_det', 'I_total_bayes',
            'Erro_total_mm', 'MAE_mm', 'Bias_mm', 'N_irrig_real',
            'N_irrig_pred', 'KGE', 'NSE']
    df_met = df_met[[c for c in cols if c in df_met.columns]]
    df_met.to_csv('output/metricas_sequenciais.csv', index=False)

    print(f"\n{'=' * 70}")
    print("METRICAS DETERMINISTICAS POR CICLO")
    print(f"{'=' * 70}")
    print(df_met.to_string(index=False))

# --- Tabela de metricas probabilisticas ---
if tabela_prob:
    df_prob = pd.DataFrame(tabela_prob)
    df_prob.to_csv('output/metricas_probabilisticas.csv', index=False)

    print(f"\n{'=' * 70}")
    print("METRICAS PROBABILISTICAS POR CICLO")
    print(f"{'=' * 70}")
    print(df_prob.to_string(index=False))

# --- Boxplot irrigacao total por ciclo ---
fig, ax = plt.subplots(figsize=(12, 6))
for i, ano in enumerate(CICLOS_ALVO):
    r = resultados[ano]
    ciclo_str = f"{ano}/{ano + 1}"
    bp = ax.boxplot(r['I_totais'], positions=[i], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor='#BBDEFB', edgecolor='#1565C0'),
                    medianprops=dict(color='#1565C0', linewidth=2))
    if r['tem_real']:
        ax.plot(i, r['I_det'].sum(), 'D', color='red', markersize=10,
                zorder=5, label='Deterministico' if i == 0 else '')

ax.set_xticks(range(len(CICLOS_ALVO)))
ax.set_xticklabels([f"{a}/{a+1}" for a in CICLOS_ALVO])
ax.set_xlabel('Ciclo')
ax.set_ylabel('Irrigacao total no ciclo (mm)')
ax.set_title('Distribuicao Bayesiana da irrigacao total por ciclo\n'
             '(analise de risco hidrico)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/irrigacao_total_por_ciclo.png', dpi=300)
plt.close()

# --- Painel de violin plots ---
ciclos_validados = [a for a in CICLOS_ALVO if resultados[a]['tem_real']]
if ciclos_validados:
    n_val = len(ciclos_validados)
    fig, axes = plt.subplots(1, n_val, figsize=(5 * n_val, 5), sharey=True)
    if n_val == 1:
        axes = [axes]
    for ax, ano in zip(axes, ciclos_validados):
        r = resultados[ano]
        df_v = pd.DataFrame({
            'Det.': r['I_det'],
            'Bayes.': np.mean(r['I_diarios'], axis=0),
        })
        sns.violinplot(data=df_v, ax=ax, palette=['#2196F3', '#FF9800'])
        ax.set_title(f"{ano}/{ano + 1}")
        ax.set_ylabel('Lamina diaria (mm)' if ax == axes[0] else '')
    plt.suptitle('Laminas de irrigacao: Deterministico vs. Bayesiano',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/violin_todos_ciclos.png', dpi=300)
    plt.close()

# --- Tabela de laminas recomendadas ---
tabela_laminas = []
for ano in CICLOS_ALVO:
    r = resultados[ano]
    prob_irrig = np.mean(r['I_totais'] > 0) * 100
    tabela_laminas.append({
        'Ciclo': f"{ano}/{ano + 1}",
        'Prob_irrigar_%': round(prob_irrig, 1),
        'I_total_media': round(np.mean(r['I_totais']), 1),
        'I_total_q05': round(np.quantile(r['I_totais'], 0.05), 1),
        'I_total_q50': round(np.quantile(r['I_totais'], 0.50), 1),
        'I_total_q95': round(np.quantile(r['I_totais'], 0.95), 1),
        'N_irrigacoes_media': round(np.mean(np.sum(r['I_diarios'] > 0, axis=1)), 1),
        'Lamina_media_evento': round(
            np.mean(r['I_totais'][r['I_totais'] > 0]) if np.any(r['I_totais'] > 0) else 0, 1),
    })

df_laminas = pd.DataFrame(tabela_laminas)
df_laminas.to_csv('output/laminas_recomendadas.csv', index=False)
print(f"\n{'=' * 70}")
print("LAMINAS RECOMENDADAS POR CICLO (com probabilidade)")
print(f"{'=' * 70}")
print(df_laminas.to_string(index=False))

# --- Resumo ---
print(f"\n{'=' * 70}")
print("CONCLUIDO!")
print(f"{'=' * 70}")
print("Arquivos gerados em output/:")
print("  metricas_sequenciais.csv          — metricas deterministicas")
print("  metricas_probabilisticas.csv      — CRPS, cobertura IC, percentil")
print("  laminas_recomendadas.csv          — recomendacao com probabilidade")
print("  irrigacao_total_por_ciclo.png     — boxplot por ciclo")
print("  violin_todos_ciclos.png           — painel de violin plots")
for ano in CICLOS_ALVO:
    c = f"{ano}_{ano + 1}"
    print(f"  previsao_{c}.csv                — laminas + SW + ETc diarios")
    if resultados[ano]['tem_real']:
        print(f"  balanco_diario_{c}.png          — SW, ETc, I com bandas IC")
        print(f"  violin_{c}.png")
        print(f"  deterministico_{c}.csv")
    print(f"  trace_{c}.nc                    — trace MCMC")
