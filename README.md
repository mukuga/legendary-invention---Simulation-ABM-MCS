# legendary-invention---Simulation-ABM-MCS

Simulasi integrasi **Decentralized Finance (DeFi)** dan perbankan konvensional menggunakan **Agent-Based Modeling (ABM)** dan **Monte Carlo Simulation (MCS)**.  
Repositori ini berisi kode, konfigurasi, serta analisis hasil simulasi berbagai skenario stres pada ekosistem keuangan.

---

## 🚀 Fitur Utama
- Implementasi **ABM** berbasis `mesa`.
- Integrasi **Monte Carlo Simulation** untuk uji ketahanan sistem.
- Mendukung berbagai **skenario shock** di sektor keuangan.
- Analisis statistik berbasis **ANOVA** dan **post-hoc tests**.
- Output dalam bentuk **CSV** dan **grafik PNG**.

---

## 📂 Struktur Proyek
```
project/
├── run.py                # Menjalankan simulasi utama
├── visualize.py          # Membuat visualisasi hasil simulasi
├── run_stats_v1p2.py     # Analisis statistik hasil simulasi (ANOVA)
├── configs/              # Parameter konfigurasi
├── results/              # Hasil output simulasi (.csv, .png)
└── agents/, model/, ...  # Modul inti simulasi
```

---

## ⚙️ Persyaratan
- **Python**: 3.11 / 3.13
- **Dependencies**:
  ```txt
  mesa==2.3.4
  numpy>=1.23
  pandas>=1.5
  scipy>=1.9
  statsmodels>=0.14
  matplotlib>=3.6
  seaborn>=0.12
  pyyaml>=6.0
  tqdm>=4.64
  pingouin>=0.5   # opsional, untuk uji post-hoc Games–Howell
  scikit-posthocs>=0.7  # alternatif untuk uji post-hoc
  ```

Install dengan:
```bash
pip install -r requirements.txt
```

---

## 📊 Skenario Simulasi
1. **S0_baseline**  
2. **S1_depeg_stablecoin**  
3. **S2_panic_selling_defi**  
4. **S3_twin_shock**  
5. **S4_twin_shock_cex_intervention**  
6. **S5_twin_shock_imf_bailout**  
7. **S6_worst_case_no_intervention**

---

## 🖥️ Cara Menjalankan
### 1. Jalankan simulasi
```bash
python run.py --scenario S0_baseline
```
Hasil berupa file `.csv` akan tersimpan di folder `results/`.

### 2. Buat visualisasi
```bash
python visualize.py --scenario S0_baseline
```
Grafik hasil simulasi akan tersimpan dalam format `.png`. 

### 3. Menjalankan langsung semua skenario secara otomatis 
```powershell
powershell -ExecutionPolicy Bypass -File .\run-all.ps1
```
Hasil berupa file hasil simulasi `.csv` dan grafik visualisasi `.png` dari seluruh skenario. 

### 4. Analisis statistik
```bash
python run_stats_v1p2.py
```
Menghasilkan output analisis **ANOVA** dan uji **post-hoc**.

--- 



## 📤 Output Utama
- **CSV**: hasil simulasi kuantitatif.  
- **PNG**: visualisasi grafik.  
- **Statistik**: hasil ANOVA & post-hoc tests.  

Contoh hasil visualisasi:

![Contoh Grafik Hasil Simulasi - stablecoin_price_timeseries]([figures/indonesia/S0_baseline/stablecoin_price_timeseries.png](https://github.com/mukuga/legendary-invention---Simulation-ABM-MCS/blob/result/figures/indonesia/S0_baseline/stablecoin_price_timeseries.png))
![Contoh Grafik Hasil Simulasi - final_price_distribution]([figures/indonesia/S0_baseline/final_price_distribution.png](https://github.com/mukuga/legendary-invention---Simulation-ABM-MCS/blob/result/figures/indonesia/S0_baseline/final_price_distribution.png))
![Contoh Grafik Hasil Simulasi - final_liquidity_boxplot]([figures/indonesia/S0_baseline/final_liquidity_boxplot.png](https://github.com/mukuga/legendary-invention---Simulation-ABM-MCS/blob/result/figures/indonesia/S0_baseline/final_liquidity_boxplot.png))

---

## 📜 Lisensi
Proyek ini dilisensikan di bawah **MIT License** – silakan lihat file [LICENSE](LICENSE) untuk detail.

---

## 🙌 Kontribusi
Pull request, issue, atau diskusi pengembangan sangat terbuka.  
Silakan fork repositori ini dan ajukan kontribusi melalui PR.

---
