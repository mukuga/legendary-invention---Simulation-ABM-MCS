import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# --- Paths ---
RESULTS_DIR = pathlib.Path('results')
FIGURES_DIR = pathlib.Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

# Load all CSV results (you can adjust pattern as needed)
csv_files = sorted(RESULTS_DIR.glob('*.csv'))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}")
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Ensure consistent column name for time step
en_col = 'Step'
if 'step' in df.columns:
    df.rename(columns={'step': en_col}, inplace=True)
elif 'Step' not in df.columns:
    df[en_col] = df.index  # fallback if no column

# --- 1) Boxplot: Likuiditas Sistem pada Akhir Simulasi ---
final_step = df[en_col].max()
liq_final = df[df[en_col] == final_step]['liquidity']
plt.figure(figsize=(8, 6))
sns.boxplot(x=liq_final)
plt.xlabel('Total Likuiditas')
plt.title('Distribusi Likuiditas Sistem pada Akhir Simulasi')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'boxplot_liquidity.png')
plt.clf()

# --- 2) Histogram + KDE: Harga Stablecoin Akhir Simulasi ---
price_final = df[df[en_col] == final_step]['price']
plt.figure(figsize=(8, 6))
sns.histplot(price_final, kde=True, stat='frequency', bins=30)
plt.xlabel('Harga Stablecoin')
plt.ylabel('Frekuensi')
plt.title('Distribusi Harga Stablecoin pada Akhir Simulasi')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'histogram_price_distribution.png')
plt.clf()

# --- 3) Time-series: Mean ± 95% CI ---
grouped = df.groupby(en_col)['price']
mean_series = grouped.mean()
lower = grouped.quantile(0.025)
upper = grouped.quantile(0.975)
plt.figure(figsize=(10, 6))
plt.plot(mean_series.index, mean_series.values, color='black', label='Mean')
plt.fill_between(mean_series.index, lower.values, upper.values, alpha=0.2, label='95% CI')
plt.xlabel('Waktu (Step)')
plt.ylabel('Harga Stablecoin')
plt.title('Simulasi Harga Stablecoin (Mean ± 95% CI)')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'timeseries_mean_CI.png')
plt.close()

# --- 4) Opsional: Sampel lintasan acak (<=30) untuk visual tambahan ---
unique_runs = df['run'].unique() if 'run' in df.columns else None
if unique_runs is not None and len(unique_runs) > 0:
    sample_runs = pd.Series(unique_runs).sample(n=min(30, len(unique_runs)), random_state=42)
    df_sample = df[df['run'].isin(sample_runs)]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sample, x=en_col, y='price', hue='run', legend=False, alpha=0.3)
    plt.plot(mean_series.index, mean_series.values, color='black', label='Mean')
    plt.xlabel('Waktu (Step)')
    plt.ylabel('Harga Stablecoin')
    plt.title('Sampel Lintasan Harga Stablecoin + Mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'timeseries_sample_runs.png')
    plt.close()

print(f"✅ Visualisasi selesai. Gambar disimpan di {FIGURES_DIR}")
