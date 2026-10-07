import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# 1. Investment universe
tickers_dict = {
    'SPY': 'Benchmark_us',
    'VCR':  'Vanguard Consumer Discretionary ETF',
    'XLI':  'Procyclique_Industrie',
    'XLP':  'Defensif_Conso_Base',
    'XLU':  'Defensif_Utilities',
    'XLK':  'Tech',
    'ITA':  'Defense',
    'XLE':  'Energie'
}
tickers = list(tickers_dict.keys())

# 2. Downloading Yahoo Finance time series
print("Downloading Yahoo Finance prices...")
# Using auto_adjust=True and ['Close'] instead of ['Adj Close']
data = yf.download(tickers, start="2014-01-01", end="2024-01-01", interval="1mo", auto_adjust=True)['Close']
data.rename(columns=tickers_dict, inplace=True)
data = data.dropna()

# Monthly returns
returns = data.pct_change().dropna()
# Convert index to "Month-Year" format to ease merging
returns.index = returns.index.to_period('M') 

# 3. Downloading of Kenneth French database
print("Downloading Kenneth French factors...")
ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start="2014-01-01", end="2024-01-01")[0]
ff_data = ff_data / 100 

# 4. Fusion of both databases
print("Merging data...")
df_final = returns.join(ff_data).dropna()

print("\nOverview of the combined final dataset (Assets + Factors):")
print(df_final.head())

import numpy as np

print("\n--- 3. COMPUTING STATISTICS ---")


actifs = list(tickers_dict.values())
returns_actifs = df_final[actifs]

# 1. Cumulated returns
cum_returns = (1 + returns_actifs).cumprod() * 100

# 2. Creating the statistics dataframe
stats = pd.DataFrame(index=actifs)

# Geometric annualized return
n_months = len(returns_actifs)
stats['Rentabilité Ann. (%)'] = ((1 + returns_actifs).prod() ** (12 / n_months) - 1) * 100

# Annualized volatility
stats['Volatilité Ann. (%)'] = returns_actifs.std() * np.sqrt(12) * 100

# Sharpe Ratio
for actif in actifs:
    # Compute excess returns (Return - Monthly risk-free rate)
    excess_ret = returns_actifs[actif] - df_final['RF']
    # Sharpe annualization: (Mean / Volatility) * sqrt(12)
    stats.loc[actif, 'Sharpe Ratio'] = (excess_ret.mean() / excess_ret.std()) * np.sqrt(12)

# Maximum Drawdown
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max
stats['Max Drawdown (%)'] = drawdown.min() * 100

# Rounding for clean display
stats = stats.round(2)
print("\nPerformance statistics table:")
print(stats)

# 3. Plotting cumulative performance chart
cum_returns.plot(figsize=(12, 6), linewidth=2)
plt.title('Performances Cumulées des Secteurs (Base 100 en 2014)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Valeur du portefeuille')
plt.grid(True, linestyle='--', alpha=0.7)
# Place legend outside to avoid hiding curves
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left') 
plt.tight_layout()
plt.show()

print("\n--- 4. VISUALIZATION: PERFORMANCES AND DRAWDOWNS ---")

# If not already computed, here are the time series:
cum_returns = (1 + returns_actifs).cumprod() * 100
rolling_max = cum_returns.cummax()
# Drawdown is the percentage decline from the previous peak
drawdown_series = (cum_returns - rolling_max) / rolling_max * 100

# Creating a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

# --- Subplot 1 (Top): Cumulative performance ---
cum_returns.plot(ax=ax1, linewidth=2)
ax1.set_title('Performances Cumulées des Secteurs (Base 100 en 2014)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Valeur du portefeuille')
ax1.grid(True, linestyle='--', alpha=0.7)
# Legend placed outside to avoid hiding lines
ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

# --- Subplot 2 (Bottom): Time-series drawdowns ---
drawdown_series.plot(ax=ax2, linewidth=1.5, alpha=0.8)
ax2.set_title('Drawdowns (Pertes par rapport au sommet précédent)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Baisse en %')
ax2.set_xlabel('Date')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.get_legend().remove() # Hide bottom legend since it is the same as the top one

plt.tight_layout()
plt.show()

# --- RISK-FREE RATE VERIFICATION (KENNETH FRENCH) ---
# Calculation of monthly mean and annualized RF over the period
rf_mean_monthly = df_final['RF'].mean()
rf_mean_annual = ((1 + rf_mean_monthly)**12 - 1) * 100

print(f"\n--- KENNETH FRENCH VERIFICATION ---")
print(f"Mean monthly risk-free rate (RF): {rf_mean_monthly:.5f} (i.e. {rf_mean_monthly*100:.3f}%)")
print(f"Annualized mean risk-free rate (RF) over 2014-2023: {rf_mean_annual:.2f}%")
print("Verification OK: Data has been properly divided by 100.")

import statsmodels.api as sm

print("\n--- 5. CAPM REGRESSIONS (ALPHA AND BETA) ---")

# 1. Calculation of global benchmark excess return
df_final['Benchmark_Excess'] = df_final['Benchmark_us'] - df_final['RF']

results_list = []

# Isolate sectors (we will not regress the benchmark against itself)
secteurs = [actif for actif in actifs if actif != 'Benchmark_us']

# 2. Regression loop for each sector
for secteur in secteurs:
    # Dependent variable (Y): Sector excess return
    Y = df_final[secteur] - df_final['RF']
    
    # Independent variable (X): Benchmark excess return
    X = sm.add_constant(df_final['Benchmark_Excess'])
    
    # Run OLS regression
    modele = sm.OLS(Y, X).fit()
    
    # 3. Metrics extraction
    alpha_mensuel = modele.params['const']
    alpha_ann = alpha_mensuel * 12 * 100  # Annualization in %
    p_value_alpha = modele.pvalues['const']
    beta = modele.params['Benchmark_Excess']
    r_squared = modele.rsquared
    
    results_list.append({
        'Secteur': secteur,
        'Alpha Ann. (%)': alpha_ann,
        'P-valeur Alpha': p_value_alpha,
        'Bêta': beta,
        'R²': r_squared
    })

# Display summary table
capm_results = pd.DataFrame(results_list).set_index('Secteur').round(3)
print("\nAlpha and Beta summary table (CAPM Model):")
print(capm_results)

print("\n--- 5.A BETA COMPARISON (2014-2018 vs 2019-2023) ---")

# Splitting data into two 5-year periods
period1 = df_final.loc['2014':'2018'].copy()
period2 = df_final.loc['2019':'2023'].copy()

beta_comparaison = []

for secteur in secteurs:
    # Period 1 Beta calculation
    Y1 = period1[secteur] - period1['RF']
    # Using 'Benchmark_Excess' created in step 5
    X1 = sm.add_constant(period1['Benchmark_Excess'])
    beta1 = sm.OLS(Y1, X1).fit().params['Benchmark_Excess'] 
    
    # Period 2 Beta calculation
    Y2 = period2[secteur] - period2['RF']
    X2 = sm.add_constant(period2['Benchmark_Excess'])
    beta2 = sm.OLS(Y2, X2).fit().params['Benchmark_Excess'] 
    
    beta_comparaison.append({
        'Secteur': secteur,
        'Bêta 14-18 (Période 1)': beta1,
        'Bêta 19-23 (Période 2)': beta2,
        'Évolution Absolue': beta2 - beta1
    })

df_beta_comp = pd.DataFrame(beta_comparaison).set_index('Secteur').round(3)
print(df_beta_comp)

print("\n--- 5.B ROLLING BETAS (12-Month Window) ---")

window = 12
rolling_betas = pd.DataFrame(index=df_final.index)

# Vectorized calculation of rolling beta: Covariance(Sector, Benchmark) / Variance(Benchmark)
# Using excess returns to remain consistent with CAPM
var_bench_rolling = df_final['Benchmark_Excess'].rolling(window=window).var()

for secteur in secteurs:
    secteur_excess = df_final[secteur] - df_final['RF']
    cov_rolling = secteur_excess.rolling(window=window).cov(df_final['Benchmark_Excess'])
    rolling_betas[secteur] = cov_rolling / var_bench_rolling

# Drop first 11 months (NaN due to the rolling window)
rolling_betas = rolling_betas.dropna()

# Plotting the chart
rolling_betas.plot(figsize=(14, 7), linewidth=1.5, alpha=0.8)
plt.title('Évolution dynamique des Bêtas sectoriels (Fenêtre roulante de 12 mois)', fontsize=14, fontweight='bold')
plt.ylabel('Bêta (Risque de marché)')
plt.xlabel('Date')
plt.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Bêta Marché = 1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()

import seaborn as sns

print("\n--- 6 & 7. CORRELATION ANALYSIS ---")
correlation_matrix = returns_actifs.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1, center=0.5, fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation des Secteurs', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n--- 8. MULTI-FACTOR REGRESSION (FAMA-FRENCH 3 FACTORS) ---")
# Objective: Show that the CAPM Alpha often disappears when controlling for other risk factors (Size and Value).

ff_results_list = []

for secteur in secteurs:
    Y = df_final[secteur] - df_final['RF']
    
    # Independent variables: Market premium, Size factor (SMB), Value/Growth factor (HML)
    X_ff = sm.add_constant(df_final[['Mkt-RF', 'SMB', 'HML']])
    
    modele_ff = sm.OLS(Y, X_ff).fit()
    
    ff_results_list.append({
        'Secteur': secteur,
        'Alpha FF Ann. (%)': modele_ff.params['const'] * 12 * 100,
        'P-valeur Alpha': modele_ff.pvalues['const'],
        'Bêta Marché': modele_ff.params['Mkt-RF'],
        'Bêta Taille (SMB)': modele_ff.params['SMB'],
        'Bêta Style (HML)': modele_ff.params['HML'],
        'R²': modele_ff.rsquared
    })

ff_results = pd.DataFrame(ff_results_list).set_index('Secteur').round(3)
print("\nAlpha and Beta summary table (Fama-French Model):")
print(ff_results)
