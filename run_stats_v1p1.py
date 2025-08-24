
#!/usr/bin/env python3
import os, glob, argparse, sys, math, json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc

REQ_COLS = {'Step','price','liquidity','iteration','scenario','region','coin'}

def load_results(root):
    paths = [p for p in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)]
    if not paths:
        raise FileNotFoundError(f'No CSV files found under: {root}')
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not REQ_COLS.issubset(df.columns):
                missing = REQ_COLS - set(df.columns)
                raise ValueError(f'Missing columns {missing} in {p}')
            frames.append(df[list(REQ_COLS)])
        except Exception as e:
            print(f'Warning: skip {p}: {e}', file=sys.stderr)
    if not frames:
        raise RuntimeError('No valid CSVs after filtering.')
    data = pd.concat(frames, ignore_index=True)
    # Normalize text columns
    for col in ['scenario','region','coin']:
        data[col] = data[col].astype(str).str.strip().str.upper()
    # Scenario base: first token (S0..S6)
    data['scenario_base'] = data['scenario'].str.split('_').str[0]
    return data

def time_to_repeg_start(s, threshold=0.99, window=12):
    price = s['price'].values
    steps = s['Step'].values
    n = len(s)
    for i in range(n):
        j = min(n, i+window)
        if np.all(price[i:j] >= threshold) and (j-i) >= min(window, n-i):
            return int(steps[i]) - int(steps[0])
    return np.nan

def time_to_repeg_from_breach(s, threshold=0.99, window=12):
    price = s['price'].values
    steps = s['Step'].values
    n = len(s)
    start_idx = None
    for i in range(n):
        if price[i] < threshold:
            start_idx = i
            break
    if start_idx is None:
        return np.nan
    for j in range(start_idx, n):
        w_end = min(n, j+window)
        if np.all(price[j:w_end] >= threshold) and (w_end-j) >= min(window, n-j):
            return int(steps[j]) - int(steps[start_idx])
    return np.nan

def liquidity_loss_pct(s):
    if s.empty: return np.nan
    L0 = s['liquidity'].iloc[0]
    L_end = s['liquidity'].iloc[-1]
    if L0 == 0: return np.nan
    return 100.0 * (L_end - L0) / L0

def compute_metrics(data, window=12, pegged=0.99, repeg_from='start'):
    metrics = []
    keys = ['region','coin','scenario','scenario_base','iteration']
    for key, g in data.groupby(keys):
        r,c,sc,scb,it = key
        g = g.sort_values('Step')
        if repeg_from == 'first_breach':
            t_rep = time_to_repeg_from_breach(g, threshold=pegged, window=window)
        else:
            t_rep = time_to_repeg_start(g, threshold=pegged, window=window)
        loss = liquidity_loss_pct(g)
        metrics.append({'region':r,'coin':c,'scenario':sc,'scenario_base':scb,'iteration':int(it),
                        't_repeg': t_rep, 'liq_loss_pct': loss})
    return pd.DataFrame(metrics)

def paired_tests(met, region, coin):
    a = met[(met.region==region)&(met.coin==coin)&(met.scenario_base=='S3')][['iteration','t_repeg']].rename(columns={'t_repeg':'S3'})
    b = met[(met.region==region)&(met.coin==coin)&(met.scenario_base=='S5')][['iteration','t_repeg']].rename(columns={'t_repeg':'S5'})
    m = pd.merge(a,b,on='iteration', how='inner').dropna()
    if len(m) < 5:
        return {'region':region,'coin':coin,'n':len(m),'t':np.nan,'p_t':np.nan,'d_paired':np.nan,'p_wilcoxon':np.nan}
    t_stat, p_val = stats.ttest_rel(m['S3'], m['S5'])
    d = t_stat / math.sqrt(len(m))
    try:
        stat, p_w = stats.wilcoxon(m['S3'], m['S5'])
    except Exception:
        p_w = np.nan
    return {'region':region,'coin':coin,'n':len(m),'t':t_stat,'p_t':p_val,'d_paired':d,'p_wilcoxon':p_w}

def anova_across_regions(met, coin, scenario_base='S5'):
    sub = met[(met.coin==coin)&(met.scenario_base==scenario_base)].dropna(subset=['liq_loss_pct'])
    groups = [g['liq_loss_pct'].values for _,g in sub.groupby('region')]
    labels = [k for k,_ in sub.groupby('region')]
    if len(groups) < 2 or any(len(g)<5 for g in groups):
        return None, None
    F, p = stats.f_oneway(*groups)
    k = len(groups); n = sum(len(g) for g in groups)
    df_b = k-1; df_w = n-k
    eta2 = (F*df_b) / (F*df_b + df_w)
    dff = pd.DataFrame({'loss':sub['liq_loss_pct'].values, 'region':sub['region'].values})
    tukey = mc.MultiComparison(dff['loss'], dff['region']).tukeyhsd()
    return (F,p,eta2,df_b,df_w), tukey

def welch_cex_vs_reg(met, region, coin):
    a = met[(met.region==region)&(met.coin==coin)&(met.scenario_base=='S4')]['liq_loss_pct'].dropna()
    b = met[(met.region==region)&(met.coin==coin)&(met.scenario_base=='S5')]['liq_loss_pct'].dropna()
    if len(a)<5 or len(b)<5:
        return {'region':region,'coin':coin,'t':np.nan,'p':np.nan,'d':np.nan,'mean_S4':a.mean() if len(a)>0 else np.nan,'mean_S5':b.mean() if len(b)>0 else np.nan,'n_S4':len(a),'n_S5':len(b)}
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean()-b.mean()) / math.sqrt(((a.var(ddof=1)+b.var(ddof=1))/2.0))
    return {'region':region,'coin':coin,'t':t_stat,'p':p_val,'d':d,'mean_S4':a.mean(),'mean_S5':b.mean(),'n_S4':len(a),'n_S5':len(b)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--threshold', type=float, default=0.99)
    ap.add_argument('--coin', default=None)
    ap.add_argument('--repeg_from', choices=['start','first_breach'], default='first_breach',
                    help='Patokan waktu re-peg: dari awal simulasi (start) atau sejak pertama kali harga < threshold (first_breach).')
    ap.add_argument('--out_dir', default='./lampiran_stats')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_results(args.results_dir)
    if args.coin:
        data = data[data['coin']==args.coin.upper()]

    met = compute_metrics(data, window=args.window, pegged=args.threshold, repeg_from=args.repeg_from)
    met.to_csv(os.path.join(args.out_dir,'metrics_per_iteration.csv'), index=False)

    regions = sorted(met['region'].unique().tolist())
    coins = sorted(met['coin'].unique().tolist())

    paired_rows = [paired_tests(met, r, c) for r in regions for c in coins]
    pd.DataFrame(paired_rows).to_csv(os.path.join(args.out_dir,'paired_S3_vs_S5_time_to_repeg.csv'), index=False)

    anova_rows = []; tukey_frames = []
    for c in coins:
        res, tukey = anova_across_regions(met, c, scenario_base='S5')
        if res is None: continue
        F,p,eta2,df_b,df_w = res
        anova_rows.append({'coin':c,'F':F,'p':p,'eta2':eta2,'df_b':df_b,'df_w':df_w})
        tk = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tk['coin'] = c
        tukey_frames.append(tk)
    pd.DataFrame(anova_rows).to_csv(os.path.join(args.out_dir,'anova_S5_liq_loss_across_regions.csv'), index=False)
    if tukey_frames:
        pd.concat(tukey_frames, ignore_index=True).to_csv(os.path.join(args.out_dir,'tukey_S5_liq_loss_across_regions.csv'), index=False)

    welch_rows = [welch_cex_vs_reg(met, r, c) for r in regions for c in coins]
    pd.DataFrame(welch_rows).to_csv(os.path.join(args.out_dir,'welch_S4_vs_S5_liq_loss.csv'), index=False)

if __name__ == '__main__':
    main()
