# =============================================================================
# MODELO DE INFERENCIA BAYESIANA SEQUENCIAL – BALANCO HIDRICO DA SOJA
# Balsas (MA) | Ciclo 90 dias | Plantio 01/12
# Atualizacao sequencial: ~60 anos -> prever 2020/21 -> atualizar -> ...
# Ref: Ribeiro (2023, 2024), FAO-56 Cap. 6, FAO-66
# Parametros da cultura: Profa. Patricia
# =============================================================================

import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

os.makedirs('output', exist_ok=True)

print("=" * 60)
print("MODELO DE INFERENCIA BAYESIANA SEQUENCIAL")
print("Soja irrigada — Balsas (MA)")
print("=" * 60)

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

# Ciclos-alvo para previsao sequencial
CICLOS_ALVO = [2020, 2021, 2022, 2023, 2024]  # ano do plantio (dezembro)

N_SAMPLES = 2000   # amostras MCMC por cadeia
N_SIM = 500        # simulacoes Monte Carlo para distribuicao de laminas


def get_kc(dia):
    """Kc por dia do ciclo (1-90). Ref: FAO-56 Cap. 6, FAO-66."""
    if   1 <= dia <= 15: return 0.40   # Inicial
    elif 16 <= dia <= 30: return 0.80  # Desenvolvimento
    elif 31 <= dia <= 70: return 1.15  # Intermediaria
    elif 71 <= dia <= 85: return 0.80  # Final
    elif 86 <= dia <= 90: return 0.50  # Colheita
    return 0.50


# =============================================================================
# 1. PREPARACAO DE DADOS
# =============================================================================

df_hist = pd.read_csv('data/Balsas_MA.csv')
df_hist['Data'] = pd.to_datetime(df_hist['Data'])
df_hist = df_hist.sort_values('Data').reset_index(drop=True)

print(f"\nDados historicos: {len(df_hist)} dias "
      f"({df_hist['Data'].dt.year.min()}-{df_hist['Data'].dt.year.max()})")


def extrair_safras(df, mes=PLANTIO_MES, dia=PLANTIO_DIA, duracao=DURACAO_CICLO):
    """Extrai janelas de 90 dias (01/12 -> ~28/02) para cada ano com dados completos."""
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
# 2. FUNCOES DE BALANCO HIDRICO (FAO-56)
# =============================================================================

def balanco_hidrico(df, awc=AWC, mad=MAD):
    """Balanco hidrico diario FAO-56. Requer colunas: dia_ciclo, ETo, pr."""
    df = df.copy()
    df['Kc'] = df['dia_ciclo'].apply(get_kc)
    df['ETc'] = df['Kc'] * df['ETo']
    df['P_eff'] = 0.8 * df['pr']

    n = len(df)
    SW = np.zeros(n)
    I_arr = np.zeros(n)
    DP = np.zeros(n)

    sw = awc  # Capacidade de campo (plantio no inicio da estacao chuvosa)
    mad_threshold = awc * (1 - mad)

    for i in range(n):
        p_eff = df.iloc[i]['P_eff']
        etc = df.iloc[i]['ETc']

        sw_new = sw + p_eff - etc

        # Drenagem profunda (excesso acima da capacidade de campo)
        dp = max(0.0, sw_new - awc)
        sw_new = min(sw_new, awc)

        # Irrigacao quando SW < limiar MAD -> retorna a capacidade de campo
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


def balanco_hidrico_deficit(deficits, awc=AWC, mad=MAD):
    """Balanco hidrico a partir de deficits diarios (ETc - P_eff).
    Retorna SW, I, DP como arrays numpy."""
    n = len(deficits)
    SW = np.zeros(n)
    I_arr = np.zeros(n)
    DP = np.zeros(n)

    sw = awc
    mad_threshold = awc * (1 - mad)

    for i in range(n):
        sw_new = sw - deficits[i]

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

    return SW, I_arr, DP


# =============================================================================
# 3. MODELO BAYESIANO E FUNCOES AUXILIARES
# =============================================================================

def treinar_modelo(df_treino, n_samples=N_SAMPLES):
    """Modelo Bayesiano hierarquico para deficit hidrico por dia do ciclo.
    Priors climatologicos derivados dos dados de treino (posteri -> prior)."""
    # Priors climatologicos
    stats = df_treino.groupby('dia_ciclo').agg(
        mu=('deficit', 'mean'),
        sigma=('deficit', 'std'),
    ).reset_index()

    with pm.Model() as model:
        mu_dia = pm.Normal(
            'mu_dia',
            mu=stats['mu'].values,
            sigma=stats['sigma'].values + 1e-5,
            shape=DURACAO_CICLO
        )

        sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)

        dias_idx = df_treino['dia_ciclo'].values - 1
        pm.Normal(
            'deficit_obs',
            mu=mu_dia[dias_idx],
            sigma=sigma_obs,
            observed=df_treino['deficit'].values
        )

        trace = pm.sample(
            draws=n_samples,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True
        )

    return model, trace


def prever_bayesiano(trace, n_sim=N_SIM, awc=AWC, mad=MAD):
    """Gera distribuicao de laminas a partir da posterior.
    Cada amostra -> gera deficits -> roda balanco hidrico."""
    mu_dia_post = trace.posterior['mu_dia'].values.reshape(-1, DURACAO_CICLO)
    sigma_post = trace.posterior['sigma_obs'].values.reshape(-1)

    n_post = mu_dia_post.shape[0]
    idx = np.random.choice(n_post, size=min(n_sim, n_post), replace=False)

    I_total_samples = []
    I_diario_samples = []

    for i in idx:
        deficit_sample = np.random.normal(mu_dia_post[i], sigma_post[i])
        _, I_sim, _ = balanco_hidrico_deficit(deficit_sample, awc, mad)
        I_total_samples.append(I_sim.sum())
        I_diario_samples.append(I_sim)

    return np.array(I_total_samples), np.array(I_diario_samples)


def calcular_metricas(y_real, y_pred):
    """KGE, NSE, MAE, Bias."""
    r = pearsonr(y_real, y_pred)[0]
    alpha = np.std(y_pred) / (np.std(y_real) + 1e-10)
    beta = np.mean(y_pred) / (np.mean(y_real) + 1e-10)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    nse = 1 - np.sum((y_real - y_pred)**2) / (np.sum((y_real - np.mean(y_real))**2) + 1e-10)
    mae = np.mean(np.abs(y_real - y_pred))
    bias = np.mean(y_pred - y_real)
    return {'KGE': round(kge, 3), 'NSE': round(nse, 3),
            'MAE': round(mae, 2), 'Bias': round(bias, 2)}


# =============================================================================
# 4. LOOP SEQUENCIAL: TREINO -> PREVISAO -> VALIDACAO -> ATUALIZACAO
#    (Ribeiro 2023/2024: posterior do ciclo N vira prior do ciclo N+1)
# =============================================================================

resultados = {}
tabela_metricas = []

for ano_alvo in CICLOS_ALVO:
    ciclo_str = f"{ano_alvo}/{ano_alvo + 1}"
    print(f"\n{'=' * 60}")
    print(f"CICLO {ciclo_str}")
    print(f"{'=' * 60}")

    # --- Dados de treino: todas as safras ANTERIORES ao ciclo-alvo ---
    df_treino = df_safras[df_safras['ano_safra'] < ano_alvo].copy()
    n_treino = df_treino['ano_safra'].nunique()
    print(f"  Treino: {n_treino} safras (ate {ano_alvo - 1}/{ano_alvo})")

    # --- Treina modelo Bayesiano ---
    print(f"  Treinando modelo Bayesiano...")
    model, trace = treinar_modelo(df_treino)
    az.to_netcdf(trace, f'output/trace_{ano_alvo}_{ano_alvo + 1}.nc')

    # --- Previsao Bayesiana ---
    I_totais, I_diarios = prever_bayesiano(trace)
    I_media_diaria = np.mean(I_diarios, axis=0)

    print(f"  Previsao: Irrigacao total = {np.mean(I_totais):.1f} mm "
          f"[IC 90%: {np.quantile(I_totais, 0.05):.1f}"
          f"-{np.quantile(I_totais, 0.95):.1f}]")

    # Salva previsao diaria
    df_pred = pd.DataFrame({
        'dia_ciclo': np.arange(1, DURACAO_CICLO + 1),
        'Kc': [get_kc(d) for d in range(1, DURACAO_CICLO + 1)],
        'I_media': I_media_diaria,
        'I_q05': np.quantile(I_diarios, 0.05, axis=0),
        'I_q25': np.quantile(I_diarios, 0.25, axis=0),
        'I_q50': np.quantile(I_diarios, 0.50, axis=0),
        'I_q75': np.quantile(I_diarios, 0.75, axis=0),
        'I_q95': np.quantile(I_diarios, 0.95, axis=0),
        'I_std': np.std(I_diarios, axis=0),
    })
    df_pred.to_csv(f'output/previsao_{ano_alvo}_{ano_alvo + 1}.csv', index=False)

    # --- Validacao com dados reais (se disponiveis) ---
    df_real = df_safras[df_safras['ano_safra'] == ano_alvo]
    tem_real = len(df_real) == DURACAO_CICLO

    if tem_real:
        df_det = balanco_hidrico(df_real.reset_index(drop=True))
        df_det.to_csv(f'output/deterministico_{ano_alvo}_{ano_alvo + 1}.csv', index=False)
        I_det = df_det['I'].values

        met = calcular_metricas(I_det, I_media_diaria)
        met['Ciclo'] = ciclo_str
        met['N_treino'] = n_treino
        met['I_total_det'] = round(I_det.sum(), 1)
        met['I_total_bayes'] = round(np.mean(I_totais), 1)
        tabela_metricas.append(met)

        print(f"  Deterministico: {I_det.sum():.1f} mm | "
              f"KGE={met['KGE']:.3f} NSE={met['NSE']:.3f} "
              f"MAE={met['MAE']:.2f} Bias={met['Bias']:.2f}")

        # --- Violin plot por ciclo ---
        fig, ax = plt.subplots(figsize=(7, 5))
        df_comp = pd.DataFrame({
            'Deterministico\n(dados reais)': I_det,
            'Bayesiano\n(previsao)': I_media_diaria,
        })
        sns.violinplot(data=df_comp, ax=ax, palette=['#2196F3', '#FF9800'])
        ax.set_ylabel('Lamina diaria (mm)')
        ax.set_title(f'Laminas de irrigacao — Ciclo {ciclo_str}\n'
                     f'KGE={met["KGE"]:.3f} | MAE={met["MAE"]:.2f} mm')
        plt.tight_layout()
        plt.savefig(f'output/violin_{ano_alvo}_{ano_alvo + 1}.png', dpi=300)
        plt.close()
    else:
        print(f"  Dados reais indisponiveis -> previsao pura (sem validacao)")

    # --- Armazena resultados ---
    resultados[ano_alvo] = {
        'I_totais': I_totais,
        'I_diarios': I_diarios,
        'I_det': I_det if tem_real else None,
        'metricas': met if tem_real else None,
        'tem_real': tem_real,
    }

# =============================================================================
# 5. SAIDAS FINAIS PARA A TESE
# =============================================================================

# --- Tabela de metricas sequenciais ---
if tabela_metricas:
    df_met = pd.DataFrame(tabela_metricas)
    cols = ['Ciclo', 'N_treino', 'I_total_det', 'I_total_bayes',
            'KGE', 'NSE', 'MAE', 'Bias']
    df_met = df_met[cols]
    df_met.to_csv('output/metricas_sequenciais.csv', index=False)
    print(f"\n{'=' * 60}")
    print("TABELA DE METRICAS SEQUENCIAIS")
    print(f"{'=' * 60}")
    print(df_met.to_string(index=False))

# --- Grafico: evolucao das metricas ao longo dos ciclos ---
if len(tabela_metricas) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ciclos = [m['Ciclo'] for m in tabela_metricas]

    for ax, metrica, cor in zip(
        axes.flat,
        ['KGE', 'NSE', 'MAE', 'Bias'],
        ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']
    ):
        valores = [m[metrica] for m in tabela_metricas]
        ax.plot(ciclos, valores, 'o-', color=cor, linewidth=2, markersize=8)
        ax.set_title(metrica, fontsize=14, fontweight='bold')
        ax.set_ylabel(metrica)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Evolucao das metricas com atualizacao sequencial Bayesiana',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/evolucao_metricas.png', dpi=300)
    plt.close()

# --- Grafico: distribuicao da irrigacao total por ciclo ---
fig, ax = plt.subplots(figsize=(12, 6))
posicoes = []
labels = []
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
    labels.append(ciclo_str)
    posicoes.append(i)

ax.set_xticks(posicoes)
ax.set_xticklabels(labels)
ax.set_xlabel('Ciclo')
ax.set_ylabel('Irrigacao total no ciclo (mm)')
ax.set_title('Distribuicao Bayesiana da irrigacao total por ciclo')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/irrigacao_total_por_ciclo.png', dpi=300)
plt.close()

# --- Tabela de laminas recomendadas por ciclo ---
tabela_laminas = []
for ano in CICLOS_ALVO:
    r = resultados[ano]
    tabela_laminas.append({
        'Ciclo': f"{ano}/{ano + 1}",
        'I_total_media': round(np.mean(r['I_totais']), 1),
        'I_total_q05': round(np.quantile(r['I_totais'], 0.05), 1),
        'I_total_q50': round(np.quantile(r['I_totais'], 0.50), 1),
        'I_total_q95': round(np.quantile(r['I_totais'], 0.95), 1),
        'N_irrigacoes_media': round(np.mean(np.sum(r['I_diarios'] > 0, axis=1)), 1),
    })

df_laminas = pd.DataFrame(tabela_laminas)
df_laminas.to_csv('output/laminas_recomendadas.csv', index=False)
print(f"\n{'=' * 60}")
print("LAMINAS RECOMENDADAS POR CICLO")
print(f"{'=' * 60}")
print(df_laminas.to_string(index=False))

# --- Painel combinado de violin plots (todos os ciclos validados) ---
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

# --- Resumo final ---
print(f"\n{'=' * 60}")
print("CONCLUIDO!")
print(f"{'=' * 60}")
print("Arquivos gerados em output/:")
print("  metricas_sequenciais.csv       — KGE, NSE, MAE, Bias por ciclo")
print("  laminas_recomendadas.csv        — irrigacao total (media + quantis)")
print("  evolucao_metricas.png           — grafico de evolucao das metricas")
print("  irrigacao_total_por_ciclo.png   — boxplot da irrigacao total")
print("  violin_todos_ciclos.png         — violin plots combinados")
for ano in CICLOS_ALVO:
    c = f"{ano}_{ano + 1}"
    print(f"  previsao_{c}.csv              — laminas diarias previstas")
    if resultados[ano]['tem_real']:
        print(f"  deterministico_{c}.csv        — balanco com dados reais")
        print(f"  violin_{c}.png                — violin plot individual")
    print(f"  trace_{c}.nc                  — trace MCMC (ArviZ)")
