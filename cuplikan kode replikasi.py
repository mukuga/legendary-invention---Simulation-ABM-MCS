# ==============================================================
# Lampiran A.4 — Kode uji statistik, diagnostik, dan replikasi
# ==============================================================

from __future__ import annotations
import os, sys, random, platform
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import multicomp
from statsmodels.stats.oneway import anova_oneway
import matplotlib.pyplot as plt

# (opsional) Games–Howell jika tersedia
try:
    import pingouin as pg   # pip install pingouin
    _HAS_PG = True
except Exception:
    _HAS_PG = False

# ---------------------------
# A.4.0 Reproducibility seed
# ---------------------------
SEED = 42  # seed global untuk seluruh analisis
def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

set_seeds()

# ---------------------------
# A.4.1 Q–Q plot residual
# ---------------------------
def qqplot_residuals_from_ols(df: pd.DataFrame, formula: str, out_path: str):
    """
    Memasang OLS sesuai formula (mis. 'y ~ C(group)') lalu membuat Q–Q plot residual.
    Gunakan untuk memeriksa normalitas residual pada uji t/ANOVA parametris.
    """
    model = smf.ols(formula, data=df).fit()
    resid = model.resid
    fig = sm.ProbPlot(resid).qqplot(line="s")
    plt.title("Q–Q Plot Residual (OLS)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return model, resid

# ---------------------------
# A.4.2 Diagnostik asumsi
# ---------------------------
def check_normality_and_levene(residuals: np.ndarray,
                               *groups: np.ndarray) -> dict:
    """
    - Shapiro–Wilk pada residual (normalitas).
    - Levene pada kelompok (homogenitas varians).
    Kembalikan p-value untuk pengambilan keputusan.
    """
    # Shapiro pada residual (untuk uji parametris)
    shapiro_p = stats.shapiro(residuals).pvalue if len(residuals) <= 5000 else \
                stats.normaltest(residuals).pvalue  # alternatif utk n besar
    # Levene antar-kelompok (varian homogen?)
    levene_p = stats.levene(*groups, center="median").pvalue
    return {"shapiro_p": shapiro_p, "levene_p": levene_p}

# ------------------------------------------------
# A.4.3 Paired test: t vs Wilcoxon (S3 vs S5, dll)
# ------------------------------------------------
def paired_test(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Uji berpasangan (t-test). Jika normalitas selisih gagal, gunakan Wilcoxon.
    """
    diff = x - y
    shapiro_p = stats.shapiro(diff).pvalue if len(diff) <= 5000 else \
                stats.normaltest(diff).pvalue

    t_stat, t_p = stats.ttest_rel(x, y)
    d_paired = t_stat / np.sqrt(len(x))  # Cohen's d (paired)

    if shapiro_p < 0.05:
        # fallback non-parametrik
        W_stat, W_p = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)
    else:
        W_stat, W_p = np.nan, np.nan

    return {"t": t_stat, "p_t": float(t_p), "d_paired": float(d_paired),
            "p_wilcoxon": (float(W_p) if not np.isnan(W_p) else None),
            "shapiro_p": float(shapiro_p)}

# ----------------------------------------------------------------------
# A.4.4 Independent t: Levene → t standar vs Welch (equal_var=False)
# ----------------------------------------------------------------------
def independent_t(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """
    Menguji homogenitas varians (Levene). Jika gagal → gunakan Welch’s t-test.
    """
    lev_p = stats.levene(group_a, group_b, center="median").pvalue
    equal_var = (lev_p >= 0.05)
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
    # Cohen's d untuk dua sampel (gunakan pooled SD; indikatif saat equal_var)
    na, nb = len(group_a), len(group_b)
    sd_pooled = np.sqrt(((na-1)*np.var(group_a, ddof=1) + (nb-1)*np.var(group_b, ddof=1)) / (na+nb-2))
    d_ind = (np.mean(group_a) - np.mean(group_b)) / sd_pooled if equal_var else None
    return {"t": float(t_stat), "p": float(p_val), "levene_p": float(lev_p),
            "equal_var": equal_var, "d": (float(d_ind) if d_ind is not None else None)}

# -------------------------------------------------------------------------
# A.4.5 Satu-arah ANOVA: standar vs Welch ANOVA + Games–Howell (opsional)
# -------------------------------------------------------------------------
def one_way_anova(groups: list[np.ndarray]) -> dict:
    """
    Lakukan Levene antar-kelompok. Jika varians homogen → ANOVA standar.
    Jika tidak homogen → Welch ANOVA (statsmodels) dan (opsional) Games–Howell.
    """
    lev_p = stats.levene(*groups, center="median").pvalue
    if lev_p >= 0.05:
        F, p, df_between, df_within = sm.stats.anova_lm(
            sm.OLS(np.concatenate(groups),
                   sm.add_constant(np.concatenate([np.repeat(i, len(g)) for i, g in enumerate(groups)]))).fit(),
            typ=2
        ).iloc[0][["F", "PR(>F)","df","df"]]  # ringkas; boleh ganti formula berbasis DataFrame
        method = "ANOVA (homogen)"
        posthoc = "Tukey HSD (statsmodels.stats.multicomp)"
    else:
        # Welch ANOVA (varian tak homogen)
        F, p, df = anova_oneway(groups, use_var="unequal")  # Welch’s F
        method = "Welch ANOVA (use_var='unequal')"
        posthoc = "Games–Howell"
    return {"F": float(F), "p": float(p), "levene_p": float(lev_p),
            "method": method, "posthoc": posthoc}

def games_howell(df: pd.DataFrame, dv: str, between: str) -> pd.DataFrame | None:
    """
    Games–Howell via pingouin (jika tersedia). Bila tidak, kembali None
    dan gunakan Tukey HSD dengan catatan interpretasi (konservatif).
    """
    if not _HAS_PG:
        return None
    return pg.pairwise_gameshowell(dv=dv, between=between, data=df)

# -------------------------------------------------------------
# A.4.6 Versi paket & lingkungan (untuk laporan replikasi)
# -------------------------------------------------------------
def print_environment():
    print("=== Environment ===")
    print("Python:", platform.python_version())
    for name, mod in [("numpy", np), ("pandas", pd), ("scipy", stats),
                      ("statsmodels", sm), ("matplotlib", plt)]:
        ver = getattr(mod, "__version__", "N/A")
        print(f"{name}=={ver}")
    print("pingouin_installed:", _HAS_PG)

print_environment()

# -------------------------------------------------------------
# A.4.7 Contoh penggunaan (S3 vs S5; ANOVA lintas-region)
# -------------------------------------------------------------
if __name__ == "__main__":
    # Misal df sudah berisi kolom: value, group (atau region), treatment (S3/S5)
    # Paired test (berpasangan per-iterasi)
    # x, y = nilai metrik S3 dan S5 yang diurut per iterasi (panjang sama)
    # t_res = paired_test(x, y)

    # Independent t (CEX vs Regulator)
    # res_ind = independent_t(group_a, group_b)

    # ANOVA (mis. min_liq_drawdown_pct pada S5 untuk tiga region)
    # groups = [df.loc[df.region=="INDONESIA","min_liq_drawdown_pct"].values,
    #           df.loc[df.region=="UNITED_STATES","min_liq_drawdown_pct"].values,
    #           df.loc[df.region=="EUROZONE","min_liq_drawdown_pct"].values]
    # res_anova = one_way_anova(groups)

    # Q–Q plot residual dari model OLS ringkas (contoh ANOVA via OLS):
    # model, resid = qqplot_residuals_from_ols(
    #     df, formula="value ~ C(region)", out_path="figures/diagnostics/qq_resid_anova.png"
    # )
    pass
