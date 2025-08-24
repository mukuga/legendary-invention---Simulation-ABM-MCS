# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse

# --- Argumen CLI ---------------------------------------------------
parser = argparse.ArgumentParser(description="Analisis hasil simulasi ABM.")
parser.add_argument("--data-dir", default="results", help="Folder tempat hasil simulasi (*.csv)")
parser.add_argument("--output-dir", default="figures", help="Folder output untuk menyimpan grafik")
args = parser.parse_args()

DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Ambil semua file hasil simulasi (*.csv) secara rekursif -------
all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
if not all_files:
    raise FileNotFoundError(f"Tidak ada file CSV ditemukan di {DATA_DIR}. Jalankan simulasi terlebih dahulu.")

# Gabungkan seluruh hasil simulasi
dfs = [pd.read_csv(f).assign(run=os.path.basename(f)) for f in all_files]
full_df = pd.concat(dfs, ignore_index=True)

# --- pastikan kolom 'Step' ada ---
if 'Step' not in full_df.columns:
    if 'step' in full_df.columns:
        full_df.rename(columns={'step': 'Step'}, inplace=True)
    else:
        full_df['Step'] = full_df.index

# Pastikan kolom numerik utama tersedia
for col in ["price", "liquidity"]:
    if col not in full_df.columns:
        raise KeyError(f"Kolom '{col}' tidak ditemukan dalam data simulasi.")

# Konversi step ke integer
full_df["Step"] = full_df["Step"].astype(int)

# --- 1. Time Series Harga -----------------------------------------
sample = full_df['run'].unique()[:50]  # ambil 50 run pertama
sns.lineplot(data=full_df[full_df['run'].isin(sample)],
             x="Step", y="price", hue="run",
             legend=False, alpha=0.3)

ci_df = full_df.groupby("Step")["price"].agg(['mean'])
ci_df['low']  = full_df.groupby("Step")["price"].quantile(0.025).values
ci_df['high'] = full_df.groupby("Step")["price"].quantile(0.975).values
plt.fill_between(ci_df.index, ci_df['low'], ci_df['high'], alpha=0.2, label="95% CI")
sns.lineplot(x=ci_df.index, y=ci_df['mean'], color="black", label="Mean")

plt.title("Simulasi Harga Stablecoin")
plt.ylabel("Harga Stablecoin")
plt.xlabel("Waktu (Step)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stablecoin_price_timeseries.png"))
plt.close()

# --- 2. Distribusi Harga Akhir -------------------------------------
last_steps = (
    full_df.loc[full_df.groupby("run")["Step"].idxmax()]
)

plt.figure(figsize=(8, 5))
sns.histplot(last_steps["price"], bins=20, kde=True)
plt.title("Distribusi Harga Stablecoin pada Akhir Simulasi")
plt.xlabel("Harga Stablecoin")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_price_distribution.png"))
plt.close()

# --- 3. Distribusi Likuiditas Sistem -------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=last_steps, y="liquidity")
plt.title("Distribusi Likuiditas Sistem pada Akhir Simulasi")
plt.ylabel("Likuiditas Tersisa (IDR)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_liquidity_boxplot.png"))
plt.close()

print("✅ Analisis selesai. Grafik disimpan di:", OUTPUT_DIR)
