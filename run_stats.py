
#!/usr/bin/env python3
import os, glob, argparse, sys, math, json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc

def load_results(root):
    paths = [p for p in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)]
    if not paths:
        raise FileNotFoundError(f'No CSV files found under: {root}')
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # Required columns
            req = {'Step','price','liquidity','iteration','scenario','region','coin'}
            if not req.issubset(df.columns):
                missing = req - set(df.columns)
                raise ValueError(f'Missing columns {missing} in {p}')
            frames.append(df[list(req)])
        except Exception as e:
            print(f'Warning: skip {p}: {e}', file=sys.stderr)
    if not frames:
        raise RuntimeError('No valid CSVs after filtering.')
    data = pd.concat(frames, ignore_index=True)
    # Normalize text columns
    data['scenario'] = data['scenario'].astype(str).str.strip().str.upper()
    data['region']   = data['region'].astype(str).str.strip().str.upper()
    data['coin']     = data['coin'].astype(str).str.strip().str.upper()
    return data

def time_to_repeg(df_iter, threshold=0.99, window=12):
    # df_iter: rows for a single (region, coin, scenario, iteration), sorted by Step
    s = df_iter.sort_values('Step')
    price = s['price'].values
    steps = s['Step'].values
    n = len(s)
    if n == 0:
        return np.nan
    # Find earliest index i such that all prices in [i, i+window-1] >= threshold
    # If window larger than remaining length, require all remaining
    for i in range(n):
        j = min(n, i+window)
        if np.all(price[i:j] >= threshold) and (j-i) >= min(window, n-i):
            return int(steps[i]) - int(steps[0])
    return np.nan

def liquidity_loss_pct(df_iter):
    s = df_iter.sort_values('Step')
    if s.empty: return np.nan
    L0 = s['liquidity'].iloc[0]
    L_end = s['liquidity'].iloc[-1]
    if L0 == 0: return np.nan
    return 100.0 * (L_end - L0) / L0

def compute_metrics(data, window=12, pegged=0.99):
    # Per (region, coin, scenario, iteration)
    keys = ['region','coin','scenario','iteration']
    metrics = []
    for (r,c,sc,it), g in data.groupby(keys):
        t_repeg = time_to_repeg(g, threshold=pegged, window=window)
        loss = liquidity_loss_pct(g)
        metrics.append({'region':r,'coin':c,'scenario':sc,'iteration':int(it),
                        't_repeg': t_repeg, 'liq_loss_pct': loss})
    met = pd.DataFrame(metrics)
    return met

def paired_tests(met, region, coin):
    # S3 vs S5, paired by iteration
    a = met[(met.region==region)&(met.coin==coin)&(met.scenario=='S3')][['iteration','t_repeg']].rename(columns={'t_repeg':'S3'})
    b = met[(met.region==region)&(met.coin==coin)&(met.scenario=='S5')][['iteration','t_repeg']].rename(columns={'t_repeg':'S5'})
    m = pd.merge(a,b,on='iteration', how='inner').dropna()
    if len(m) < 5:
        return {'region':region,'coin':coin,'n':len(m),'test':'paired t','t':np.nan,'p':np.nan,'effect_d':np.nan}
    t_stat, p_val = stats.ttest_rel(m['S3'], m['S5'])
    d = t_stat / math.sqrt(len(m))
    # Wilcoxon alternative
    try:
        z_stat, p_w = stats.wilcoxon(m['S3'], m['S5'])
        r_eff = z_stat / math.sqrt(len(m))
    except Exception:
        p_w, r_eff = np.nan, np.nan
    return {'region':region,'coin':coin,'n':len(m),'test':'paired t (wilcoxon alt)',
            't':t_stat,'p_t':p_val,'d_paired':d,'p_wilcoxon':p_w,'r_wilcoxon':r_eff}

def anova_across_regions(met, coin, scenario='S5'):
    sub = met[(met.coin==coin)&(met.scenario==scenario)].dropna(subset=['liq_loss_pct'])
    groups = [g['liq_loss_pct'].values for _,g in sub.groupby('region')]
    labels = [k for k,_ in sub.groupby('region')]
    if len(groups) < 2 or any(len(g)<5 for g in groups):
        return None, None, None
    F, p = stats.f_oneway(*groups)
    # eta squared
    # Build full arrays
    vals = np.concatenate(groups)
    # Calculate df
    k = len(groups)
    n = len(vals)
    df_b = k-1
    df_w = n-k
    eta2 = (F*df_b) / (F*df_b + df_w)
    # Tukey
    dff = pd.DataFrame({'loss':sub['liq_loss_pct'].values, 'region':sub['region'].values})
    tukey = mc.MultiComparison(dff['loss'], dff['region']).tukeyhsd()
    return (F,p,eta2,df_b,df_w), tukey, dff

def ttest_cex_vs_regulator(met, region, coin):
    a = met[(met.region==region)&(met.coin==coin)&(met.scenario=='S4')]['liq_loss_pct'].dropna()
    b = met[(met.region==region)&(met.coin==coin)&(met.scenario=='S5')]['liq_loss_pct'].dropna()
    if len(a)<5 or len(b)<5:
        return {'region':region,'coin':coin,'test':'Welch t','t':np.nan,'p':np.nan,'d':np.nan}
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's d for independent samples (Hedges' g optional)
    d = (a.mean()-b.mean()) / math.sqrt(((a.var(ddof=1)+b.var(ddof=1))/2.0))
    return {'region':region,'coin':coin,'test':'Welch t','t':t_stat,'p':p_val,'d':d,
            'mean_S4':a.mean(),'mean_S5':b.mean(),'n_S4':len(a),'n_S5':len(b)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True, help='Directory containing CSV results (recursively).')
    ap.add_argument('--window', type=int, default=12, help='Consecutive-step window for re-peg detection (default 12).')
    ap.add_argument('--threshold', type=float, default=0.99, help='Price threshold for re-peg (default 0.99).')
    ap.add_argument('--coin', default=None, help='Filter by coin symbol (e.g., USDC/EURC). Default: all.')
    ap.add_argument('--out_dir', default='./lampiran_stats', help='Output directory for summary tables.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_results(args.results_dir)
    if args.coin:
        data = data[data['coin'].str.upper()==args.coin.upper()]
    # Compute metrics
    met = compute_metrics(data, window=args.window, pegged=args.threshold)
    met.to_csv(os.path.join(args.out_dir,'metrics_per_iteration.csv'), index=False)

    # Paired tests S3 vs S5 (time-to-repeg) per region
    regions = sorted(met['region'].unique().tolist())
    coins = sorted(met['coin'].unique().tolist())
    paired_rows = []
    for r in regions:
        for c in coins:
            paired_rows.append(paired_tests(met, r, c))
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(args.out_dir,'paired_S3_vs_S5_time_to_repeg.csv'), index=False)

    # ANOVA across regions for S5 liquidity loss (per coin)
    anova_rows = []
    tukey_frames = []
    for c in coins:
        res, tukey, dff = anova_across_regions(met, c, scenario='S5')
        if res is None: 
            continue
        F,p,eta2,df_b,df_w = res
        anova_rows.append({'coin':c,'F':F,'p':p,'eta2':eta2,'df_b':df_b,'df_w':df_w})
        tk = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tk['coin'] = c
        tukey_frames.append(tk)
    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_csv(os.path.join(args.out_dir,'anova_S5_liq_loss_across_regions.csv'), index=False)
    if tukey_frames:
        tukey_df = pd.concat(tukey_frames, ignore_index=True)
        tukey_df.to_csv(os.path.join(args.out_dir,'tukey_S5_liq_loss_across_regions.csv'), index=False)

    # Welch t-test S4 vs S5 liquidity loss (per region & coin)
    welch_rows = []
    for r in regions:
        for c in coins:
            welch_rows.append(ttest_cex_vs_regulator(met, r, c))
    welch_df = pd.DataFrame(welch_rows)
    welch_df.to_csv(os.path.join(args.out_dir,'welch_S4_vs_S5_liq_loss.csv'), index=False)

    # Print quick summaries
    print('Saved outputs to:', os.path.abspath(args.out_dir))
    print('Files:\n - metrics_per_iteration.csv\n - paired_S3_vs_S5_time_to_repeg.csv\n - anova_S5_liq_loss_across_regions.csv\n - tukey_S5_liq_loss_across_regions.csv\n - welch_S4_vs_S5_liq_loss.csv')

if __name__ == '__main__':
    main()
