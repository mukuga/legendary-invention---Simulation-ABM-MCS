# ---------------------------------------------------------------------------
# agents.py
# ---------------------------------------------------------------------------
from __future__ import annotations

from mesa import Agent
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from model import DeFiBankModel


def clip(x: float, lo: float, hi: float) -> float:
    """Utility to keep ratios within [lo, hi]."""
    return min(max(x, lo), hi)


class BankAgent(Agent):
    """Commercial bank exposed to DeFi‑linked assets."""

    def __init__(
        self,
        unique_id: int,
        model: "DeFiBankModel",
        capital: float,
        crypto_exposure: float,
        interest_rate: float,
        exposure_threshold: float = 0.97,  # price at which write‑down starts
    ) -> None:
        super().__init__(unique_id, model)
        self.capital = capital
        self.crypto_exposure = crypto_exposure  # $ value of stablecoin holdings
        self.interest_rate = interest_rate
        self.exposure_threshold = exposure_threshold

    # ------------------------------------------------------------------
    #  Core behaviours
    # ------------------------------------------------------------------
    def lend(self, borrower: "BorrowerAgent", requested: float) -> None:
        """Issue loan up to 10 % of current capital."""
        loan_amount = min(requested, self.capital * 0.10)
        if loan_amount <= 0:
            return
        self.capital -= loan_amount
        borrower.receive_loan(loan_amount, self.interest_rate)

    def mark_to_market(self) -> None:
        """Write‑down crypto exposure if price below threshold."""
        price = self.model.market.S
        if price < self.exposure_threshold:
            loss_ratio = (self.exposure_threshold - price) / self.exposure_threshold
            loss = self.crypto_exposure * loss_ratio
            self.capital -= loss
            self.crypto_exposure *= price / self.exposure_threshold  # new value

    def step(self) -> None:  # executed by Mesa scheduler
        self.mark_to_market()
        # lending is initiated by BorrowerAgent


class BorrowerAgent(Agent):
    """Household/firm choosing between bank and DeFi borrowing."""

    def __init__(self, unique_id: int, model: "DeFiBankModel", *,
                    risk_tolerance: float, loan_demand: float, 
                    default_prob: float = 0.02) -> None:
        super().__init__(unique_id, model)
        self.risk_tolerance = risk_tolerance
        self.loan_demand    = loan_demand
        self.default_prob   = default_prob
        self.outstanding_loan = 0.0

    # ------------------------------------------------------------------
    def choose_lender(self) -> str:
        price = self.model.market.S
        defi_yield = price * np.random.uniform(0.05, 0.15)
        bank_rates = [a.interest_rate for a in self.model.banks]
        bank_rate = min(bank_rates) if bank_rates else np.inf
        risk_adjusted_yield = defi_yield - (1 - price) * (1 - self.risk_tolerance)
        return "defi" if risk_adjusted_yield > bank_rate else "bank"

    def receive_loan(self, amount: float, rate: float) -> None:
        self.outstanding_loan += amount * (1 + rate)
        self.loan_demand = max(0.0, self.loan_demand - amount)

    def maybe_default(self) -> None:
        if self.outstanding_loan <= 0:
            return
        panic_p = getattr(self.model, 'panic_prob_this_step', 0.0)
        shock_mult = getattr(self.model, 'default_shock_mult', 1.0)
        prob = self.default_prob * (1 + panic_p) * shock_mult
        if np.random.rand() < prob:
            self.outstanding_loan = 0.0
            self.model.register_default()

    def step(self) -> None:
        # 1) borrowing decision (only if demand remains)
        if self.loan_demand > 0:
            choice = self.choose_lender()
            if choice == "bank" and self.model.banks:
                best_bank = min(self.model.banks, key=lambda b: b.interest_rate)
                best_bank.lend(self, self.loan_demand)
            else:
                defi_rate = self.model.market.S * np.random.uniform(0.05, 0.15)
                self.receive_loan(self.loan_demand, defi_rate)
        # 2) possible default
        self.maybe_default()


class ArbitrageurAgent(Agent):
    """Trader exploiting stable‑coin peg deviations."""

    def __init__(self, unique_id: int, model: "DeFiBankModel", capital: float) -> None:
        super().__init__(unique_id, model)
        self.capital = capital

    def step(self) -> None:
        spread = abs(1.0 - self.model.market.S)

        # abaikan deviasi kecil (<3 %)
        if spread <= 0.03:
            return

        # batasi spread maksimum 50 % untuk mencegah profit astronomis
        effective_spread = min(spread, 0.50)

        # batasi volume – pakai min(capital*0.5, 1 juta)
        volume = min(self.capital * 0.5, 1_000_000)

        profit = volume * effective_spread
        self.capital += profit

        # dampak pasar kecil → mendekatkan harga ke peg
        self.model.market.S += 0.05 * (1.0 - self.model.market.S) * (volume / 1_000_000)


class CEXAgent(Agent):
    """Centralised Exchange back‑stopping the peg via reserves."""

    def __init__(self, unique_id: int, model: "DeFiBankModel", reserve: float) -> None:
        super().__init__(unique_id, model)
        self.reserve = reserve

    def intervene(self, magnitude: float = 0.25) -> None:
        if self.reserve <= 0:
            return
        injection = self.reserve * magnitude
        self.reserve -= injection
        self.model.market.S += injection * 0.01
        # add liquidity to banks proportionally
        boost = injection * 0.5 / max(len(self.model.banks), 1)
        for b in self.model.banks:
            b.capital += boost

    def step(self) -> None:
        pass  # reactive only via scenarios


class RegulatorAgent(Agent):
    """IMF‑style lender of last resort."""

    def __init__(
        self,
        unique_id: int,
        model: "DeFiBankModel",
        liquidity_buffer: float,
    ) -> None:
        super().__init__(unique_id, model)
        self.liquidity_buffer = liquidity_buffer
        self.initial_buffer   = liquidity_buffer   # <- NEW: for usage metric

    def intervene(self, magnitude: float = 0.20) -> None:
        if self.liquidity_buffer <= 0:
            return
        boost = self.liquidity_buffer * magnitude
        self.liquidity_buffer -= boost
        # distribute boost to banks directly
        share = boost / max(len(self.model.banks), 1)
        for b in self.model.banks:
            b.capital += share
        # minor peg support as confidence effect
        self.model.market.S += boost * 0.015

    def step(self) -> None:
        pass