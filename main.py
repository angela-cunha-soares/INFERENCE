# =============================================================================
# MODELO DE INFERENCIA BAYESIANA SEQUENCIAL – BALANCO HIDRICO DA SOJA
# Balsas (MA) | Ciclo 90 dias | Plantio 01/12
#
# AVANCOS SOBRE RIBEIRO (2024):
#   1. Balanco hidrico diario FAO-56 (SW, Kc por fase, CAD, raiz)
#   2. Reamostragem climatologica historica (60+ anos de dados reais)
#   3. Classificacao via SPEI (Vicente-Serrano et al., 2010)
#      + Metodo dos Quantis (Pinkayan 1966, Xavier et al. 2002)
#      + Indice de Aridez IA = P/ETo (UNEP 1992)
#   4. Cenarios com dados diarios REAIS (preserva veranicos e correlacoes)
#   5. ATUALIZACAO SEQUENCIAL REAL: posterior Dirichlet -> prior proximo ciclo
#      (conjugacy Dir-Multinomial: alpha_t+1 = alpha_t + counts_t)
#   6. AR(1) temporal preservado implicitamente nas sequencias reais
#   7. Laminas de irrigacao com distribuicao completa + decisao otima
#   8. Validacao dia a dia + metricas probabilisticas
#
# Ref: Ribeiro (2024), FAO-56 Cap. 6, FAO-66, Day (1985) ESP
#      Vicente-Serrano et al. (2010) SPEI, UNEP (1992) Aridez
# Parametros da cultura: Profa. Patricia
# =============================================================================

import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, fisk, norm
import warnings
warnings.filterwarnings("ignore")

os.makedirs('output', exist_ok=True)

print("=" * 70)
print("MODELO DE INFERENCIA BAYESIANA SEQUENCIAL")
print("Reamostragem climatologica historica — Soja irrigada, Balsas (MA)")
print("Continuidade: Ribeiro (2024) + FAO-56 + anos secos/normais/umidos")
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
# 3. MODELO BAYESIANO – REAMOSTRAGEM CLIMATOLOGICA HISTORICA
#    Classificacao via SPEI (Standardized Precipitation-Evapotranspiration Index)
#    + Metodo dos Quantis para definir limiares (seco/normal/umido)
#    + Indice de Aridez (IA = P/ETo) como caracterizacao complementar
#    Cenarios com dados diarios REAIS (preserva veranicos e correlacoes)
#    Pesos Bayesianos via Dirichlet-Multinomial
#    ATUALIZACAO SEQUENCIAL REAL: posterior_t -> prior_{t+1}
#
#    Ref: Vicente-Serrano et al. (2010) — SPEI
#         Pinkayan (1966), Xavier et al. (2002) — Quantis
#         UNEP (1992) — Indice de Aridez
#         Day (1985) — ESP
# =============================================================================

def calcular_spei(d_series):
    """Calcula SPEI a partir da serie de D = P - ETo por safra.

    Procedimento (Vicente-Serrano et al., 2010):
    1. D = P_total - ETo_total para cada safra
    2. Ajusta distribuicao log-logistica (fisk) aos valores de D
    3. Transforma para distribuicao Normal padrao -> SPEI

    Como D pode ser negativo, aplica-se deslocamento (shift) para
    garantir valores positivos antes do ajuste log-logistico.
    """
    d = np.array(d_series, dtype=float)

    # Deslocamento para valores positivos (necessario para log-logistica)
    shift = 0.0
    if d.min() <= 0:
        shift = abs(d.min()) + 1.0
    d_pos = d + shift

    # Ajuste da distribuicao log-logistica (fisk em scipy)
    params = fisk.fit(d_pos)
    cdf_values = fisk.cdf(d_pos, *params)

    # Evita extremos (0 e 1) que gerariam infinitos
    cdf_values = np.clip(cdf_values, 1e-6, 1 - 1e-6)

    # Transforma para Normal padrao -> SPEI
    spei = norm.ppf(cdf_values)
    return spei


def treinar_modelo(df_treino, alpha_prior=None, n_samples=N_SAMPLES):
    """Modelo Bayesiano com reamostragem climatologica historica.

    Classificacao de anos via SPEI + Quantis:
    - SPEI (Vicente-Serrano et al., 2010): indice padronizado que incorpora
      P e ETo, sensivel a mudancas climaticas
    - Metodo dos Quantis (Pinkayan 1966, Xavier et al. 2002): classificacao
      direta baseada na distribuicao estatistica da serie
    - Indice de Aridez IA = P/ETo (UNEP 1992): caracterizacao complementar

    Classificacao por Quantis do SPEI:
      Seco:   SPEI <= quantil(1/3)
      Normal: quantil(1/3) < SPEI <= quantil(2/3)
      Umido:  SPEI > quantil(2/3)

    Modelo Dirichlet-Multinomial com atualizacao sequencial:
      Prior:     Dir(alpha_1, alpha_2, alpha_3)
      Dados:     (n_seco, n_normal, n_umido) contagens observadas
      Posterior: Dir(alpha_1 + n_seco, alpha_2 + n_normal, alpha_3 + n_umido)
      => Posterior do ciclo t vira prior do ciclo t+1

    Ref: Vicente-Serrano et al. (2010), Day (1985), Gelman et al. (2013)
    """
    # Prior: uniforme no primeiro ciclo, ou posterior do ciclo anterior
    if alpha_prior is None:
        alpha_prior = np.ones(3)
    alpha_prior = np.array(alpha_prior, dtype=float)

    # --- Calcular D = P - ETo e SPEI por safra ---
    stats_safra = df_treino.groupby('ano_safra').agg(
        P_total=('pr', 'sum'),
        ETo_total=('ETo', 'sum')
    ).reset_index()
    stats_safra['D'] = stats_safra['P_total'] - stats_safra['ETo_total']
    stats_safra['IA'] = stats_safra['P_total'] / stats_safra['ETo_total']
    stats_safra['SPEI'] = calcular_spei(stats_safra['D'].values)

    # --- Classificar por Quantis do SPEI ---
    q33_spei = stats_safra['SPEI'].quantile(1/3)
    q67_spei = stats_safra['SPEI'].quantile(2/3)

    categorias = {}
    for _, row in stats_safra.iterrows():
        ano = int(row['ano_safra'])
        spei = row['SPEI']
        if spei <= q33_spei:
            categorias[ano] = 0   # seco
        elif spei <= q67_spei:
            categorias[ano] = 1   # normal
        else:
            categorias[ano] = 2   # umido

    nomes_cat = ['Seco', 'Normal', 'Umido']
    counts = np.array([
        sum(1 for c in categorias.values() if c == k) for k in range(3)
    ])

    # --- Extrair sequencias diarias reais por categoria ---
    sequencias = {0: [], 1: [], 2: []}
    for ano in sorted(categorias.keys()):
        cat = categorias[ano]
        df_ano = df_treino[df_treino['ano_safra'] == ano].reset_index(drop=True)
        row_stats = stats_safra[stats_safra['ano_safra'] == ano].iloc[0]
        sequencias[cat].append({
            'ano': ano,
            'ETo': df_ano['ETo'].values.copy(),
            'pr': df_ano['pr'].values.copy(),
            'P_total': row_stats['P_total'],
            'ETo_total': row_stats['ETo_total'],
            'D': row_stats['D'],
            'SPEI': row_stats['SPEI'],
            'IA': row_stats['IA'],
        })

    # --- Indice de Aridez medio (UNEP 1992) ---
    ia_medio = stats_safra['IA'].mean()
    # IA > 0.65 = umido, 0.50-0.65 = sub-umido seco, 0.20-0.50 = semiarido
    if ia_medio > 0.65:
        classe_ia = 'Umido'
    elif ia_medio > 0.50:
        classe_ia = 'Sub-umido seco'
    elif ia_medio > 0.20:
        classe_ia = 'Semiarido'
    else:
        classe_ia = 'Arido'

    print(f"  --- Classificacao SPEI + Quantis ---")
    print(f"  SPEI limiares: q33={q33_spei:.2f}, q67={q67_spei:.2f}")
    for k, nome in enumerate(nomes_cat):
        if sequencias[k]:
            speis = [s['SPEI'] for s in sequencias[k]]
            p_tots = [s['P_total'] for s in sequencias[k]]
            d_vals = [s['D'] for s in sequencias[k]]
            print(f"    {nome}: {counts[k]} safras | "
                  f"SPEI medio={np.mean(speis):.2f} | "
                  f"D medio={np.mean(d_vals):.0f} mm | "
                  f"P medio={np.mean(p_tots):.0f} mm")
    print(f"  Indice de Aridez (IA = P/ETo): {ia_medio:.2f} ({classe_ia})")
    print(f"  Prior Dirichlet: alpha = [{alpha_prior[0]:.1f}, {alpha_prior[1]:.1f}, {alpha_prior[2]:.1f}]")

    # --- Modelo Bayesiano: Dirichlet-Multinomial com prior sequencial ---
    with pm.Model() as model:
        # Prior Dirichlet: uniforme no 1o ciclo, posterior anterior nos demais
        weights = pm.Dirichlet('weights', a=alpha_prior)

        # Likelihood: contagens observadas nas 3 categorias
        pm.Multinomial('obs', n=int(counts.sum()), p=weights,
                       observed=counts)

    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.90,
            return_inferencedata=True,
            progressbar=True
        )

    # Posterior analitico: alpha_posterior = alpha_prior + counts
    # (conjugacy Dirichlet-Multinomial — usado como prior do proximo ciclo)
    alpha_posterior = alpha_prior + counts

    # Diagnostico MCMC
    print(f"  Diagnostico MCMC:")
    summary = az.summary(trace, var_names=['weights'])
    summary.index = [f'P({n})' for n in nomes_cat]
    print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']].to_string())
    print(f"  Posterior Dirichlet: alpha = [{alpha_posterior[0]:.1f}, {alpha_posterior[1]:.1f}, {alpha_posterior[2]:.1f}]")

    # --- AR(1) empirico preservado nas sequencias reais ---
    rhos_pr = []
    for cat_seqs in sequencias.values():
        for s in cat_seqs:
            pr = s['pr']
            if len(pr) > 1 and np.std(pr) > 0:
                rhos_pr.append(np.corrcoef(pr[:-1], pr[1:])[0, 1])
    rho_empirico = np.mean(rhos_pr) if rhos_pr else 0.0
    print(f"  AR(1) empirico da precipitacao (rho): {rho_empirico:.3f} "
          f"(preservado nas sequencias reais)")

    return model, trace, sequencias, alpha_posterior


def prever_bayesiano(trace, sequencias, n_sim=N_SIM, awc=AWC, mad=MAD):
    """Previsao por reamostragem de safras historicas com pesos Bayesianos.

    Para cada simulacao:
    1. Amostra pesos (P_seco, P_normal, P_umido) da posterior Dirichlet
    2. Sorteia categoria com base nos pesos
    3. Sorteia aleatoriamente uma safra historica daquela categoria
    4. Usa os dados diarios REAIS de ETo e P daquela safra
    5. Roda balanco hidrico FAO-56 -> irrigacao com veranicos reais

    Resultado: distribuicao de irrigacao baseada em cenarios climaticos reais.
    """
    weights_post = trace.posterior['weights'].values.reshape(-1, 3)
    n_post = weights_post.shape[0]

    I_total_samples = []
    I_diario_samples = []
    SW_samples = []
    ETc_samples = []

    for i in range(n_sim):
        # Amostra pesos da posterior Dirichlet
        idx = np.random.randint(n_post)
        w = weights_post[idx]

        # Amostra categoria (seco/normal/umido)
        cat = np.random.choice(3, p=w)

        # Amostra uma safra historica daquela categoria
        anos_cat = sequencias[cat]
        safra = anos_cat[np.random.randint(len(anos_cat))]

        # Balanco hidrico com dados diarios REAIS
        SW_sim, I_sim, _, ETc_sim = balanco_hidrico_arrays(
            safra['ETo'], safra['pr']
        )

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
alpha_atual = None  # Prior inicial: Dirichlet(1,1,1) — nao-informativo

for ano_alvo in CICLOS_ALVO:
    ciclo_str = f"{ano_alvo}/{ano_alvo + 1}"
    print(f"\n{'=' * 70}")
    print(f"CICLO {ciclo_str}")
    print(f"{'=' * 70}")

    # Treino: todas as safras anteriores ao ciclo-alvo
    df_treino = df_safras[df_safras['ano_safra'] < ano_alvo].copy()
    n_treino = df_treino['ano_safra'].nunique()
    print(f"  Treino: {n_treino} safras (ate {ano_alvo - 1}/{ano_alvo})")

    # Treina modelo com prior sequencial (posterior do ciclo anterior)
    print(f"  Classificando safras e treinando modelo Bayesiano...")
    model, trace, sequencias, alpha_atual = treinar_modelo(
        df_treino, alpha_prior=alpha_atual
    )
    az.to_netcdf(trace, f'output/trace_{ano_alvo}_{ano_alvo + 1}.nc')

    # Previsao Bayesiana por reamostragem historica
    I_totais, I_diarios, SW_sims, ETc_sims = prever_bayesiano(trace, sequencias)
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
