# run.py  –  Patched 2025‑07‑29
"""Jalankan simulasi ABM‑MCS paralel untuk seluruh konfigurasi YAML.

Contoh:
    python run.py --pattern 'config_*.yml'
"""

import argparse, multiprocessing as mp, os, glob, pandas as pd
from config import load_config
from model import DeFiBankModel

def run_single(iteration: int, config_path: str):
    cfg = load_config(config_path)

    # --- DEBUG: tampilkan parameter pasar sekali per iterasi -------------
    print(f"[Iter {iteration}] market_params → {cfg['market_params']}")

    # Jaga re-producibility tapi tetap unik:
    base_seed = cfg.get("seed", 42)
    cfg["seed"] = base_seed + iteration          # <— kunci utama

    model = DeFiBankModel(cfg)
    model.run_model(cfg['steps'])
    df = model.datacollector.get_model_vars_dataframe()
    df['iteration'] = iteration
    df['scenario']  = cfg['metadata'].get('scenario')
    df['region']    = cfg['metadata'].get('region')
    df['coin']      = cfg['metadata'].get('coin')
    df.attrs["seed"] = cfg["seed"]        # simpan untuk audit
    return df


# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pattern", default="config_*.yml",
                        help="Glob pattern YAML konfigurasi.")
    parser.add_argument("--config", action="append",              # ← harus ditambahkan SEBELUM parser.parse_args()
                        help="Path YAML konfigurasi tunggal. Bisa dipanggil berkali-kali.")
    parser.add_argument("--outdir", default="results")

    args = parser.parse_args()

    # Ini baru boleh setelah argumen di atas terdaftar
    if args.config:
        config_files = [os.path.abspath(p) for p in args.config]
    else:
        config_files = sorted(glob.glob(args.pattern))


    # Pastikan folder hasil tersedia
    os.makedirs(args.outdir, exist_ok=True)

    # Beban paralel ditentukan oleh n_processes di config pertama
    base_cfg = load_config(config_files[0])
    pool = mp.Pool(processes=base_cfg['n_processes'])

    # Buat tuple list (iter, path) untuk seluruh config × n_iterations
    jobs = []
    for cfg_path in config_files:
        cfg = load_config(cfg_path)
        jobs.extend([(i, cfg_path) for i in range(cfg['n_iterations'])])

    for df in pool.starmap(run_single, jobs):
        scen = df['scenario'].iloc[0]
        it   = df['iteration'].iloc[0]
        fname = f"{args.outdir}/{scen}_iter{it}.csv"
        df.to_csv(fname, index=False)

    print(f"✅  Selesai {len(jobs)} simulasi → {args.outdir}/")
