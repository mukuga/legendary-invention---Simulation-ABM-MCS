# ---------------------------------------------------------------------------
# market.py
# ---------------------------------------------------------------------------
import numpy as np
from typing import Dict

# --- parameter global (atas file) ---------------------------------
MIN_PRICE, MAX_PRICE = 0.05, 2.0         # hard-floor & hard-cap
KAPPA = 0.05                             # kecepatan mean-reversion → 0 = off
# ------------------------------------------------------------------

class Market:
    """Geometric Brownian Motion + Jump‑Diffusion for stable‑coin price."""

    def __init__(
        self,
        mu: float,
        sigma: float,
        jump_prob: float,
        jump_size: list[float],
        shock_schedule: Dict[int, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.S: float = 1.0  # peg
        self.mu, self.sigma = mu, sigma
        self.jump_prob, self.jump_size = jump_prob, jump_size
        # seed numpy RNG untuk market ini (opsional)
        if seed is not None:
            np.random.seed(seed)
        self.shock_schedule = shock_schedule or {}
        self.step_count: int = 0

    # ------------------------------------------------------------------
    def update_price(self) -> None: 
        """GBM + jump + mean-reversion + clipping untuk stabilitas numerik."""
        dt = 1
        dW = np.random.normal(0, np.sqrt(dt))

        # 1) Geometric Brownian Motion
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        self.S *= np.exp(drift + self.sigma * dW)

        # 2) Jump-diffusion
        if np.random.rand() < self.jump_prob:
            self.S *= (1 + np.random.choice(self.jump_size))

        # 3) Mean-reversion ringan ke peg = 1
        self.S += KAPPA * (1.0 - self.S)

        # 4) Shock terjadwal
        if self.step_count in self.shock_schedule:
            self.inject_shock(self.shock_schedule[self.step_count])

        # 5) Clipping agar tidak runaway
        self.S = np.clip(self.S, MIN_PRICE, MAX_PRICE)

        self.step_count += 1
    

    def inject_shock(self, magnitude: float) -> None:
        self.S *= (1 - magnitude)
        self.S = np.clip(self.S, 0.05, 5.0)
    
    def is_peg_broken(self, tolerance: float = 0.02) -> bool:
        return abs(self.S - 1.0) > tolerance

    def force_depeg(self, severity: float = 0.5) -> None:
        self.S *= (1.0 - severity)
        self.S = np.clip(self.S, MIN_PRICE, MAX_PRICE)
