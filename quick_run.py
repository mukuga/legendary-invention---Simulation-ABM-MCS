"""
Quick smoke test – 1 iter × 200 steps
"""
import os, pathlib
from param_loader import dump_yaml
from config import load_config
from model import DeFiBankModel
import pandas as pd

# 1) Build config.yml
dump_yaml()                                # ✔️  config.yml generated

# 2) Load config & build model
conf = load_config("config.yml")
m = DeFiBankModel(conf, rng_seed=conf["seed"])

# 3) Run 200 steps
m.run_model(200)

# 4) Save output
pathlib.Path("results").mkdir(exist_ok=True)
df = m.datacollector.get_model_vars_dataframe()
df.to_csv("results/quick_test.csv", index=False)
print(df.tail())
print("🚀  Quick test finished → results/quick_test.csv")
