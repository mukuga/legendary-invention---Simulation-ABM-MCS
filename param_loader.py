"""
Reads the five JSON config files, validates SHA‑256 checksum
(if checksums.txt is present), and emits a single dict that
Mesa’s DeFiBankModel understands.
"""
from pathlib import Path
import json, hashlib, yaml

RAW_DIR = Path("Dataset")  # adjust if needed
CFG_FILES = [
    "agent_parameters.json",
    "stablecoin_market.json",
    "macro_context.json",
    "monte_carlo_config.json",
    "scenario_templates.json",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def validate_checksums():
    chkfile = RAW_DIR / "checksums.txt"
    if not chkfile.exists():
        print("⚠ checksums.txt not found – skipping integrity check")
        return
    expected = {line.split()[1]: line.split()[0] for line in chkfile.read_text().splitlines()}
    for fname in CFG_FILES:
        calc = sha256(RAW_DIR / fname)
        if fname in expected and expected[fname] != calc:
            raise ValueError(f"Checksum mismatch for {fname}")

def load() -> dict:
    validate_checksums()
    jp = lambda fn: json.loads((RAW_DIR / fn).read_text(encoding="utf-8"))
    agent, market, macro, mcs, scen = map(jp, CFG_FILES)

    # --- choose stable‑coin profile (here: USDC) ------------------------
    coin_profile = market["volatility_profile"]["USDC"]
    market_params = {
        "mu": coin_profile["mu"],
        "sigma": coin_profile["sigma"],
        "jump_prob": coin_profile["jump_prob"],
        "jump_size": coin_profile["jump_size"],
        "shock_schedule": {},  # filled by scenario loader
    }

    # --- transform agent parameters (only one bank & borrower demo) ----
    banks = [{
        "class": "BankAgent",
        "id": 1,
        "params": {
            "capital": agent["bank"]["initial_capital"]["Indonesia"]["value"],
            "crypto_exposure": (agent["bank"]["crypto_exposure_ratio"]["value"]
                                * agent["bank"]["initial_capital"]["Indonesia"]["value"]),
            "interest_rate": agent["bank"]["interest_rate"]["Indonesia"]["value"],
        },
    }]
    borrowers = [{
        "class": "BorrowerAgent",
        "id": 10,
        "params": {
            "risk_tolerance": 0.6,
            "loan_demand": 25_000_000,  # mid‑range
            "default_prob": agent["borrower"]["default_probability"]["base"],
        },
    }]
    others = [
        {"class": "ArbitrageurAgent", "id": 20,
         "params": {"capital": agent["arbitrageur"]["initial_capital"]["value"]}},
        {"class": "CEXAgent", "id": 30,
         "params": {"reserve": agent["cex"]["initial_reserve"]["value"]}},
        {"class": "RegulatorAgent", "id": 40,
         "params": {"liquidity_buffer": agent["regulator"]["liquidity_buffer"]["value"]}},
    ]

    cfg = {
        "seed": mcs["run_config"]["random_seed"],
        "steps": mcs["run_config"]["steps_per_iteration"],
        "n_iterations": mcs["run_config"]["n_iterations"],
        "n_processes": mcs["run_config"]["n_processes"],
        "panic_threshold": 0.95,               # default; overridden per‑scenario
        "market_params": market_params,
        "agents": banks + borrowers + others,
        "scenarios": {},                       # filled below
    }
    # ---------------------------------------------------------------

    # --- choose which scenario template to run --------------------------
    ACTIVE_SCEN = "S0_baseline"          # ⇠ ubah via CLI/argparser nanti
    scen_dict = jp("scenario_templates.json")[ACTIVE_SCEN]
    cfg["scenarios"] = scen_dict                 # flattened!
    
    return cfg

def dump_yaml(path: str = "config.yml"):
    cfg = load()
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("✅  config.yml generated")

if __name__ == "__main__":
    dump_yaml()
