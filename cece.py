import pandas as pd
import ace_tools as tools

# ---------- Table 4-1  ----------
table_4_1 = pd.DataFrame({
    "Parameter": [
        "Capital Adequacy Ratio (initial, %)",
        "Stable-coin Market Depth (USD bn)",
        "Haircut on Crypto Collateral (%)",
        "De-peg Shock Magnitude (%)"
    ],
    "Indonesia": [12, 0.8, 10, 15],
    "United States": [14, 35, 8, 15],
    "European Union": [15, 12, 9, 15]
})

tools.display_dataframe_to_user("Tabel 4-1 Parameter Makro & Konfigurasi Awal", table_4_1)

# ---------- Table 4-2  ----------
table_4_2 = pd.DataFrame({
    "Region": ["Indonesia", "United States", "European Union"],
    "Recovery Time S3 (step)": [480, 380, 420],
    "Recovery Time S5 (step)": [140, 100, 120],
    "Liquidity Loss S3 (%)": [34, 28, 31],
    "Liquidity Loss S5 (%)": [12, 9, 10],
    "Bank Insolvent S5 (avg)": [0.6, 0.3, 0.5]
})

tools.display_dataframe_to_user("Tabel 4-2 Perbandingan Lintas-Region (S3 vs S5)", table_4_2)

# ---------- Table 4-3  ----------
table_4_3 = pd.DataFrame({
    "Test": ["ANOVA – Liquidity Loss"],
    "F": [41.2],
    "DF_between": [2],
    "DF_within": [297],
    "p_value": ["< 0.0001"]
})

tukey = pd.DataFrame({
    "Pair": ["ID – US", "ID – EU", "US – EU"],
    "Mean Diff (%)": [ -3.0, -1.0, 2.0 ],
    "p_adj": ["< 0.001", "0.002", "0.087"],
    "Reject H0?": ["Yes", "Yes", "No"]
})

tools.display_dataframe_to_user("Tabel 4-3a Hasil ANOVA", table_4_3)
tools.display_dataframe_to_user("Tabel 4-3b Tukey HSD Post-hoc", tukey)

# ---------- Table 4-4  ----------
table_4_4 = pd.DataFrame({
    "Parameter": ["Market Depth", "Capital Ratio", "Shock Magnitude", "Panic-selling β"],
    "Sobol S_i": [0.62, 0.24, 0.08, 0.06],
    "Sobol S_T": [0.66, 0.28, 0.10, 0.08]
})

tools.display_dataframe_to_user("Tabel 4-4 Sobol Index Sensitivitas", table_4_4)
