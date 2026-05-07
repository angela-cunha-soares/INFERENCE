"""Converter CSV de resultados para tabela LaTeX."""
import pandas as pd

# Ler o CSV
df = pd.read_csv('output/forecast_sequential/sequential_summary_all_cities.csv')

# Selecionar colunas principais para não ficar muito largo
cols_principais = [
    'city', 'cycle', 'observed_class',
    'det_I_total_obs_mm', 'det_I_total_forecast_median_mm', 'det_I_total_error_mm',
    'det_MAE_SW_mm', 'det_KGE_SW',
    'prob_CRPS_I_total_mm', 'prob_PIT_I_total', 'prob_coverage_90_I_total',
    'CRPSS_vs_naive_climatology'
]

df_reduced = df[cols_principais].copy()

# Renomear colunas para formato mais legível em LaTeX
rename_dict = {
    'city': 'City',
    'cycle': 'Cycle',
    'observed_class': 'Class',
    'det_I_total_obs_mm': '$I_{\mathrm{obs}}$ (mm)',
    'det_I_total_forecast_median_mm': '$I_{\mathrm{fcst,50}}$ (mm)',
    'det_I_total_error_mm': 'Error (mm)',
    'det_MAE_SW_mm': 'MAE$_{\theta}$ (mm)',
    'det_KGE_SW': 'KGE$_{\theta}$',
    'prob_CRPS_I_total_mm': 'CRPS (mm)',
    'prob_PIT_I_total': 'PIT',
    'prob_coverage_90_I_total': 'Cov$_{90}$',
    'CRPSS_vs_naive_climatology': 'CRPSS'
}

df_reduced = df_reduced.rename(columns=rename_dict)

# Arredondar valores
numeric_cols = df_reduced.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df_reduced[col] = df_reduced[col].round(2)

# Gerar LaTeX com longtable
latex_str = "\\begin{longtable}{lrrrrrrrrrrrr}\n"
latex_str += "\\caption{Climatological sequential-forecast results across all 50 city $\\times$ cycle combinations (2020/21--2024/25). "
latex_str += "Columns: City; Cycle; SPEI tercile class (0=dry, 1=normal, 2=wet); Observed and forecast irrigation depths; "
latex_str += "Error and soil-water metrics; CRPS (Continuous Ranked Probability Score); PIT (Probability-Integral-Transform); "
latex_str += "90\\% credible-interval coverage; CRPSS (skill score against naive climatology).}\\\\ \n"
latex_str += "\\label{tab:sequential_all_cities}\\\\ \n"

# Headers
latex_str += "\\toprule\n"
latex_str += " & ".join(df_reduced.columns) + " \\\\ \n"
latex_str += "\\midrule\n"
latex_str += "\\endhead\n"

# Rows
for _, row in df_reduced.iterrows():
    row_str = " & ".join([str(val) for val in row.values])
    latex_str += row_str + " \\\\ \n"

latex_str += "\\bottomrule\n"
latex_str += "\\end{longtable}\n"

# Salvar em arquivo
with open('output/paper_tables/table_sequential_summary_all_cities.tex', 'w') as f:
    f.write(latex_str)

print("✓ Tabela LaTeX criada: output/paper_tables/table_sequential_summary_all_cities.tex")
print(f"Total de linhas de dados: {len(df_reduced)}")
print(f"Total de colunas: {len(df_reduced.columns)}")
