#!/usr/bin/env python
"""
run_all.py – orkestrator sekali jalan
-------------------------------------
• Bangun YAML per region & skenario
• Jalankan run.py → results/<region>/<scenario>
• Buat grafik → figures/<region>/<scenario>
"""

import subprocess, os, pathlib, itertools, argparse, sys

REGIONS   = ["indonesia", "united_states", "european_union"]
SCENARIOS = [
    "S0_baseline", "S1_depeg_stablecoin", "S2_panic_selling_defi",
    "S3_twin_shock", "S4_twin_shock_cex_intervention",
    "S5_twin_shock_imf_bailout", "S6_worst_case_no_intervention"
]
COIN = "USDC"

def sh(cmd: list[str]):
    """Jalankan perintah child; forward stdout/stderr."""
    print("▶", " ".join(cmd))
    res = subprocess.run(cmd, check=True)
    return res

def main(n_iter: int | None):
    root = pathlib.Path(".").resolve()
    for region, scen in itertools.product(REGIONS, SCENARIOS):
        # ---------- 1) Bangun YAML ----------
        cfg_name = f"config_{scen}_{COIN}_{region}.yml"
        cfg_path = root / cfg_name
        cmd_build = [
            sys.executable, "param_loader_2.py",
            "--scenario", scen,
            "--region", region,
            "--coin", COIN,
            "--out", str(cfg_path)
        ]
        if n_iter:                 # override iterasi utk smoke-test
            cmd_build += ["--n-iter", str(n_iter)]
        sh(cmd_build)

        # ---------- 2) Jalankan simulasi ----------
        outdir = root / "results" / region / scen
        outdir.mkdir(parents=True, exist_ok=True)
        sh([
            sys.executable, "run.py",
            "--config", str(cfg_path),
            "--outdir", str(outdir)
        ])

    # ---------- 3) Buat grafik per folder ----------
    for region, scen in itertools.product(REGIONS, SCENARIOS):
        data_dir = root / "results" / region / scen
        if not any(data_dir.glob("*.csv")):
            print("⚠️  Skip visualisasi – tidak ada CSV:", data_dir)
            continue
        out_dir = root / "figures" / region / scen
        out_dir.mkdir(parents=True, exist_ok=True)
        sh([
            sys.executable, "Visualisasi.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(out_dir)
        ])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int,
                    help="Jalankan <n> iterasi saja untuk uji cepat")
    args = ap.parse_args()
    main(args.smoke)
