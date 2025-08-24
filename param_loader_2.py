#!/usr/bin/env python3
"""
param_loader_2.py  –  Patched 2025‑07‑30
----------------------------------------------------------
Membangun konfigurasi YAML unified untuk simulasi DeFi–Bank
Tambahan patch:
    • REGION_ALIAS untuk sinkronisasi penamaan wilayah
    • Akses data melalui alias agar menghindari KeyError/TypeError
    • CLI --region kini menerima alias maupun nama kanonik
"""

import argparse
import json
import random
import yaml
import pathlib
import sys
import copy
from typing import Dict, List

# -----------------------------------------------------------------------
# Berkas pengaturan kanonik
# -----------------------------------------------------------------------

CFG_FILES = {
    "agents": "Dataset/agent_parameters.json",
    "macro": "Dataset/macro_context.json",
    "market": "Dataset/stablecoin_market.json",
    "scen": "Dataset/scenario_templates.json",
    "mcs": "Dataset/monte_carlo_config.json",
}

def jp(f: str):  # json‑parse helper
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)

# -----------------------------------------------------------------------
# Region alias mapping — sinkronisasi kunci data & CLI
# -----------------------------------------------------------------------
REGION_ALIAS = {
    "eu": "eurozone",
}

# FX rates to IDR (rough mid‑2025)
FX = {"IDR": 1.0, "USD": 16_000.0, "EUR": 17_500.0}

# -----------------------------------------------------------------------
# Helper sampling utilities
# -----------------------------------------------------------------------

def sample_uniform(dist_cfg: Dict[str, float], rng: random.Random) -> float:
    """Sample float uniformly dari [`min`,`max`] menggunakan RNG deterministik."""
    return rng.uniform(dist_cfg["min"], dist_cfg["max"])

# -----------------------------------------------------------------------
# Agent‑generation routines
# -----------------------------------------------------------------------

def build_agents(
    agent_cfg: dict,
    macro_ctx: dict,
    region: str,
    n_banks: int,
    n_borrowers: int,
    rng: random.Random,
) -> List[dict]:
    """Return list of agent‑spec dictionaries untuk DeFiBankModel."""

    region_key = REGION_ALIAS.get(region, region)

    banks: List[dict] = []
    borrowers: List[dict] = []

    # --- Banks ---------------------------------------------------------
    cap_info = agent_cfg["bank"]["initial_capital"][region_key]
    bank_cap_idr = cap_info["value"] * FX.get(cap_info.get("unit", "IDR"), 1)
    crypto_ratio = agent_cfg["bank"]["crypto_exposure_ratio"]["value"]
    interest = agent_cfg["bank"]["interest_rate"][region_key]["value"]

    for i in range(n_banks):
        banks.append(
            {
                "class": "BankAgent",
                "id": i + 1,
                "params": {
                    "capital": bank_cap_idr,
                    "crypto_exposure": bank_cap_idr * crypto_ratio,
                    "interest_rate": interest,
                },
            }
        )

    # --- Borrowers -----------------------------------------------------
    br_cfg = agent_cfg["borrower"]
    rt_dist = {
        "min": br_cfg["risk_tolerance_distribution"]["min"],
        "max": br_cfg["risk_tolerance_distribution"]["max"],
    }
    ld_range = br_cfg["loan_demand_range"]["value"]
    dp_base = br_cfg["default_probability"]["base"]
    dp_mult = br_cfg["default_probability"]["shock_multiplier_range"][1]
    dp_range = {"min": dp_base, "max": dp_base * dp_mult}

    for j in range(n_borrowers):
        borrowers.append(
            {
                "class": "BorrowerAgent",
                "id": 10_000 + j,
                "params": {
                    "risk_tolerance": sample_uniform(rt_dist, rng),
                    "loan_demand": sample_uniform({"min": ld_range[0], "max": ld_range[1]}, rng),
                    "default_prob": sample_uniform(dp_range, rng),
                },
            }
        )

    # --- Systemic singletons ------------------------------------------
    others = [
        {
            "class": "ArbitrageurAgent",
            "id": 20,
            "params": {"capital": agent_cfg["arbitrageur"]["initial_capital"]["value"]},
        },
        {
            "class": "CEXAgent",
            "id": 30,
            "params": {"reserve": agent_cfg["cex"]["initial_reserve"]["value"]},
        },
        {
            "class": "RegulatorAgent",
            "id": 40,
            "params": {
                "liquidity_buffer": macro_ctx["regulator_liquidity_buffer"]["value"],
            },
        },
    ]

    return banks + borrowers + others


def build_market(market_cfg: dict, coin: str) -> dict:
    profile = market_cfg["volatility_profile"]
    if coin not in profile:
        raise ValueError(f"Stablecoin '{coin}' tidak ditemukan dalam volatility_profile.")
    base = profile[coin]
    return {
        "mu": base["mu"],
        "sigma": base["sigma"],
        "jump_prob": base["jump_prob"],
        "jump_size": base["jump_size"],
    }

# -----------------------------------------------------------------------
# Config builder
# -----------------------------------------------------------------------

def build_config(args: argparse.Namespace, scenario_key: str) -> dict:
    agent = jp(CFG_FILES["agents"])
    market = jp(CFG_FILES["market"])
    mcs = jp(CFG_FILES["mcs"])
    scen_tpl = jp(CFG_FILES["scen"])
    macro = jp(CFG_FILES["macro"])

    rng = random.Random(args.seed)

    # Sinkronisasi alias region
    region_key = REGION_ALIAS.get(args.region, args.region)

    cfg = {
        "seed": args.seed,
        "steps": mcs["run_config"]["steps_per_iteration"],
        "n_iterations": mcs["run_config"]["n_iterations"],
        "n_processes": mcs["run_config"]["n_processes"],
        "panic_threshold": 0.95,
        "market_params": build_market(market, args.coin),
        "agents": build_agents(
            agent,
            macro[region_key],
            region_key,
            args.n_banks,
            args.n_borrowers,
            rng,
        ),
        "scenarios": scen_tpl[scenario_key],
        "stochastic": mcs.get("stochastic_parameters", {}),
        "metadata": {
            "scenario": scenario_key,
            "coin": args.coin,
            "region": region_key,  # simpan kunci kanonik
            "generated": True,
        },
    }
    return cfg

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    # Choices untuk skenario & coin tetap
    parser.add_argument("--scenario", default="S0_baseline", choices=jp(CFG_FILES["scen"]).keys())
    parser.add_argument("--coin", default="USDC", choices=jp(CFG_FILES["market"])["volatility_profile"].keys())

    # Region kini mendukung alias
    region_choices = sorted(set(jp(CFG_FILES["macro"]).keys()) | set(REGION_ALIAS.keys()))
    parser.add_argument(
        "--region",
        default="indonesia",
        choices=region_choices,
        help="Macroeconomic context region (alias ok)",
    )

    parser.add_argument("--n-banks", type=int, default=3)
    parser.add_argument("--n-borrowers", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="config.yml", help="Output YAML file path (ignored when --all-scenarios)")
    parser.add_argument("--all-scenarios", action="store_true", help="Generate YAML untuk seluruh template skenario")

    args = parser.parse_args()

    if args.all_scenarios:
        for sk in jp(CFG_FILES["scen"]).keys():
            cfg = build_config(args, sk)
            out_path = f"config_{sk}_{args.coin}_{args.region}.yml"
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"✅  Generated {out_path}")
    else:
        cfg = build_config(args, args.scenario)
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"✅  Generated {args.out}")

if __name__ == "__main__":
    main()
