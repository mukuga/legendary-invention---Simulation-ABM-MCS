from __future__ import annotations

import numpy as np, random, math
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from typing import List, Dict, Any
import scenarios
from market import Market
from agents import BankAgent, BorrowerAgent, ArbitrageurAgent, CEXAgent, RegulatorAgent

AGENT_CLASSES = {cls.__name__: cls for cls in (
    BankAgent, BorrowerAgent, ArbitrageurAgent, CEXAgent, RegulatorAgent)
}

class DeFiBankModel(Model):
    """Core ABM integrating DeFi & legacy banks + Monte‑Carlo stochastic sampling."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        seed = config.get("seed", 42)
        self.rand = np.random.default_rng(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Scheduler
        self.schedule = RandomActivation(self)
        self.current_step = 0

        # Panic default trackers
        self.in_panic: bool = False
        self.panic_prob_this_step: float = 0.0
        self._defaults_this_step: int = 0

        # Market -----------------------------------------------------------
        mp = config["market_params"]
        self.market = Market(mu=mp["mu"], sigma=mp["sigma"],
                             jump_prob=mp["jump_prob"], jump_size=mp["jump_size"])

        # Build agents -----------------------------------------------------
        for spec in config["agents"]:
            cls = AGENT_CLASSES[spec["class"]]
            agent = cls(spec["id"], self, **spec["params"])
            self.schedule.add(agent)

        self.banks: List[BankAgent] = [a for a in self.schedule.agents if isinstance(a, BankAgent)]
        self.borrowers: List[BorrowerAgent] = [a for a in self.schedule.agents if isinstance(a, BorrowerAgent)]

        # ---------------------------------------------------------------
        # STOCHASTIC SAMPLER  (Monte‑Carlo support)
        # ---------------------------------------------------------------
        self.sampled: Dict[str, float] = {}

        def _sample(spec: Dict[str, Any]) -> float:
            dist = spec.get("distribution")
            if dist == "beta":
                x = random.betavariate(spec["alpha"], spec["beta"])
                lo, hi = spec.get("range", [0.0, 1.0])
                return lo + x * (hi - lo)
            elif dist == "uniform":
                lo, hi = spec["range"]
                return random.random() * (hi - lo) + lo
            elif dist == "triangular":
                return random.triangular(spec["min"], spec["max"], spec["mode"])
            elif dist == "normal":
                return random.gauss(spec["mean"], spec["std_dev"])
            elif dist == "poisson":
                # λ mewakili frekuensi per‑step dalam konteks kita
                return spec["lambda"]
            else:
                return spec.get("value", 0.0)

        for k, v in config.get("stochastic", {}).items():
            self.sampled[k] = _sample(v)

        # Propagate to model & market ------------------------------------
        self.market.jump_prob = self.sampled.get("stablecoin_jump_frequency", self.market.jump_prob)
        self.default_shock_mult = self.sampled.get("default_probability_shock_multiplier", 1.0)
        self.shock_injection_magnitude = self.sampled.get("shock_injection_magnitude")

        # Loss counters
        self.total_loss = 0.0
        self.counterfactual_loss = 1.0

        # Data collection --------------------------------------------------
        # Log kolom yang dipakai di Visualisasi dan pastikan kapitalisasi Step
        self.step_count = 0
        self.datacollector = DataCollector(
            model_reporters={
                "Step":      lambda m: m.step_count,
                "price":     lambda m: m.market.S,
                "liquidity": lambda m: sum(b.capital for b in m.banks),
            },
            tables={},
            agent_reporters={}
        )

    # ------------------------------------------------------------------
    def register_default(self):
        self._defaults_this_step += 1

    # ------------------------------------------------------------------
    def step(self):
        self._defaults_this_step = 0

        # Example dynamic panic beta (if provided separately)
        if "panic_selling_probability" in self.config.get("stochastic", {}):
            beta = self.config["stochastic"]["panic_selling_probability"]
            self.panic_prob_this_step = self.rand.beta(beta["alpha"], beta["beta"])
        else:
            self.panic_prob_this_step = 0.0

        self.schedule.step()       # 1) agent actions
        self.market.update_price() # 2) market dynamics
        scenarios.run_all(self)    # 3) scenarios / shocks

        # Record current step and log model vars
        self.step_count += 1
        self.datacollector.collect(self)

        self.current_step += 1      # 4) advance time

    # ------------------------------------------------------------------
    def run_model(self, steps: int):
        for _ in range(steps):
            self.step()
