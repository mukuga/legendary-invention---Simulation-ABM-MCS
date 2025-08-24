#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_stats_v1p2.py — Transient Metrics for Hybrid Finance Sim Outputs
-------------------------------------------------------------------
Inputs
  --results_dir  : root folder containing raw step-level CSVs (recursive)
  --out_dir      : output folder (created if missing)
  --threshold    : price threshold for peg (default 0.99)
  --window       : consecutive steps for stable re-peg (default 12)
  --repeg_from   : "start" (from t0) or "first_breach" (from first time price<threshold)
  --coin         : optional filter, e.g. USDC/EURC (case-insensitive)

CSV schema required in raw files:
  Step, price, liquidity, iteration, scenario, region, coin

Outputs
  metrics_transient_per_iteration.csv
  A9_paired_S3_vs_S5_time_under_threshold.csv
  A10_anova_S5_min_drawdown_across_regions.csv
  A10_tukey_S5_min_drawdown_across_regions.csv
  A11_welch_S4_vs_S5_area_under_liquidity.csv
"""
import os, glob, argparse, sys, math
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc

REQ_COLS = {'Step','price','liquidity','iteration','scenario','region','coin'}

def canonical_scenario(name:str)->str:
    s = str(name).upper()
    for base in ["S0","S1","S2","S3","S4","S5","S6"]:
        if s.startswith(base):
            return base
    return s

def load_results(root):
    paths = [p for p in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)]
    if not paths:
        raise FileNotFoundError(f'No CSV files found under: {root}')
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not REQ_COLS.issubset(df.columns):
                continue
            df = df[list(REQ_COLS)].copy()
            df['scenario'] = df['scenario'].astype(str).apply(canonical_scenario)
            df['region']   = df['region'].astype(str).str.strip().str.upper().replace({'EU':'EUROZONE','US':'UNITED_STATES','ID':'INDONESIA'})
            df['coin']     = df['coin'].astype(str).str.strip().str.upper()
            frames.append(df)
        except Exception as e:
            print(f'Warning: skip {p}: {e}', file=sys.stderr)
    if not frames:
        raise RuntimeError('No valid CSVs after filtering.')
    data = pd.concat(frames, ignore_index=True)
    return data

def time_to_repeg(series_price, series_step, threshold=0.99, window=12, from_mode='first_breach'):
    n = len(series_price)
    if n == 0: return np.nan
    if from_mode not in ('first_breach','start'):
        from_mode = 'first_breach'
    start_idx = 0
    if from_mode == 'first_breach':
        under = np.where(series_price < threshold)[0]
        if len(under) == 0:
            return 0
        start_idx = int(under[0])
    for i in range(start_idx, n):
        j = min(n, i+window)
        if np.all(series_price[i:j] >= threshold) and (j-i) >= min(window, n-i):
            return int(series_step[i]) - int(series_step[start_idx])
    return np.nan

def min_drawdown_pct(series_liq):
    if len(series_liq)==0: return np.nan
    L0 = float(series_liq[0])
    if L0 == 0: return np.nan
    Lmin = float(np.min(series_liq))
    return 100.0 * (Lmin/L0 - 1.0)

def area_under_liquidity_pct(series_liq):
    if len(series_liq)==0: return np.nan
    L0 = float(series_liq[0])
    if L0 == 0: return np.nan
    avg = float(np.mean(series_liq))
    return 100.0 * (avg/L0 - 1.0)

def time_under_threshold(series_price, threshold=0.99):
    if len(series_price)==0: return np.nan
    return int(np.sum(np.array(series_price) < threshold))

def compute_metrics(data, threshold=0.99, window=12, from_mode='first_breach'):
    keys = ['region','coin','scenario','iteration']
    rows = []
    for (r,c,sc,it), g in data.groupby(keys):
        g = g.sort_values('Step')
        price = g['price'].values
        step  = g['Step'].values
        liq   = g['liquidity'].values
        met = {
            'region': r, 'coin': c, 'scenario': sc, 'iteration': int(it),
            't_repeg': time_to_repeg(price, step, threshold, window, from_mode),
            'time_under_threshold': time_under_threshold(price, threshold),
            'min_liq_drawdown_pct': min_drawdown_pct(liq),
            'area_under_liquidity_pct': area_under_liquidity_pct(liq),
        }
        rows.append(met)
    return pd.DataFrame(rows)

def paired_test(met, region, metric_col, sA='S3', sB='S5', coin=None):
    q = (met['region']==region) & (met['scenario'].isin([sA,sB]))
    if coin: q &= (met['coin'].str.upper()==coin.upper())
    sub = met.loc[q, ['scenario','iteration',metric_col,'coin']].copy()
    a = sub[sub['scenario']==sA][['iteration',metric_col]].rename(columns={metric_col:'A'})
    b = sub[sub['scenario']==sB][['iteration',metric_col]].rename(columns={metric_col:'B'})
    m = a.merge(b, on='iteration').dropna()
    if len(m) < 5:
        return {'region': region, 'metric': metric_col, 'n': len(m), 't': np.nan, 'p_t': np.nan, 'd_paired': np.nan, 'p_wilcoxon': np.nan}
    t_stat, p_val = stats.ttest_rel(m['A'], m['B'])
    d = t_stat / math.sqrt(len(m))
    try:
        W, p_w = stats.wilcoxon(m['A'], m['B'])
    except Exception:
        p_w = np.nan
    return {'region': region, 'metric': metric_col, 'n': len(m), 't': t_stat, 'p_t': p_val, 'd_paired': d, 'p_wilcoxon': p_w}

def anova_regions(met, metric_col, scenario='S5', coin=None):
    q = (met['scenario']==scenario)
    if coin: q &= (met['coin'].str.upper()==coin.upper())
    sub = met.loc[q, ['region', metric_col]].dropna()
    groups = [g[metric_col].values for _, g in sub.groupby('region')]
    labels = [k for k,_ in sub.groupby('region')]
    if len(groups) < 2 or any(len(g)<5 for g in groups):
        return None, None
    F, p = stats.f_oneway(*groups)
    k = len(groups); n = sum(len(g) for g in groups)
    df_b = k-1; df_w = n-k
    eta2 = (F*df_b) / (F*df_b + df_w)
    comp = mc.MultiComparison(sub[metric_col].values, sub['region'].values)
    tk = comp.tukeyhsd()
    tukey_df = pd.DataFrame(data=tk._results_table.data[1:], columns=tk._results_table.data[0])
    return pd.DataFrame([{'metric':metric_col,'F':F,'p':p,'eta2':eta2,'df_b':df_b,'df_w':df_w}]), tukey_df

def welch_s4_vs_s5(met, metric_col, region, coin=None):
    q4 = (met['region']==region) & (met['scenario']=='S4')
    q5 = (met['region']==region) & (met['scenario']=='S5')
    if coin:
        q4 &= (met['coin'].str.upper()==coin.upper())
        q5 &= (met['coin'].str.upper()==coin.upper())
    a = met.loc[q4, metric_col].dropna()
    b = met.loc[q5, metric_col].dropna()
    if len(a)<5 or len(b)<5:
        return {'region':region,'metric':metric_col,'n_S4':len(a),'n_S5':len(b),'t':np.nan,'p':np.nan,'d':np.nan,'mean_S4':a.mean(),'mean_S5':b.mean()}
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean()-b.mean()) / math.sqrt(((a.var(ddof=1)+b.var(ddof=1))/2.0))
    return {'region':region,'metric':metric_col,'n_S4':len(a),'n_S5':len(b),'t':t_stat,'p':p_val,'d':d,'mean_S4':a.mean(),'mean_S5':b.mean()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--out_dir', default='./lampiran_stats_v1p2')
    ap.add_argument('--threshold', type=float, default=0.99)
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--repeg_from', choices=['start','first_breach'], default='first_breach')
    ap.add_argument('--coin', default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_results(args.results_dir)
    if args.coin:
        data = data[data['coin'].str.upper()==args.coin.upper()]

    met = compute_metrics(data, threshold=args.threshold, window=args.window, from_mode=args.repeg_from)
    met.to_csv(os.path.join(args.out_dir, 'metrics_transient_per_iteration.csv'), index=False)

    regions = sorted(met['region'].unique().tolist())

    a9_rows = [paired_test(met, r, 'time_under_threshold', 'S3','S5', args.coin) for r in regions]
    pd.DataFrame(a9_rows).to_csv(os.path.join(args.out_dir, 'A9_paired_S3_vs_S5_time_under_threshold.csv'), index=False)

    a10_anova, a10_tukey = anova_regions(met, 'min_liq_drawdown_pct', scenario='S5', coin=args.coin)
    if a10_anova is not None:
        a10_anova.to_csv(os.path.join(args.out_dir, 'A10_anova_S5_min_drawdown_across_regions.csv'), index=False)
        a10_tukey.to_csv(os.path.join(args.out_dir, 'A10_tukey_S5_min_drawdown_across_regions.csv'), index=False)

    a11_rows = [welch_s4_vs_s5(met, 'area_under_liquidity_pct', r, args.coin) for r in regions]
    pd.DataFrame(a11_rows).to_csv(os.path.join(args.out_dir, 'A11_welch_S4_vs_S5_area_under_liquidity.csv'), index=False)

    print('Saved outputs to:', os.path.abspath(args.out_dir))
    print('Files:')
    print(' - metrics_transient_per_iteration.csv')
    print(' - A9_paired_S3_vs_S5_time_under_threshold.csv')
    print(' - A10_anova_S5_min_drawdown_across_regions.csv')
    print(' - A10_tukey_S5_min_drawdown_across_regions.csv')
    print(' - A11_welch_S4_vs_S5_area_under_liquidity.csv')

if __name__ == '__main__':
    main()
