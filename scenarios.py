from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from model import DeFiBankModel

# ---------------------------------------------------------------------
# Loader for reading dynamic scenario flags once (step == 0)
# ---------------------------------------------------------------------

def load_dynamic_flags(model: "DeFiBankModel"):
    cfg = model.config.get("scenarios", {})
    # Setup panic selling flags
    model.s_panic = cfg.get("panic_selling", {}).get("active", False)
    model.panic_threshold = cfg.get("panic_selling", {}).get("threshold_price", 0.95)
    model.panic_multiplier_cfg = cfg.get("panic_selling", {}).get("panic_multiplier", 5.0)
    model.in_panic = False
    model.panic_multiplier = 1.0

# ---------------------------------------------------------------------
# Individual scenario functions – called every step
# ---------------------------------------------------------------------

def scenario_depeg(model: "DeFiBankModel") -> None:
    cfg = model.config.get("scenarios", {}).get("depeg", {})
    # Only apply if active
    if not cfg.get("active", False):
        return
    # New preferred key
    severity = cfg.get("severity")
    # Fallback for legacy templates using magnitude
    if severity is None:
        severity = cfg.get("magnitude")
    # Ensure a shock magnitude is provided
    if severity is None:
        raise ValueError("De-peg severity not specified")
    # Inject shock
    if model.current_step == cfg.get("step", 0):
        model.market.inject_shock(severity)


def scenario_panic(model: "DeFiBankModel") -> None:
    cfg = model.config.get("scenarios", {}).get("panic_selling", {})
    shock_step = cfg.get("shock_step")
    # If shock_step is specified, trigger panic at that step
    if shock_step is not None:
        if model.current_step == shock_step and not model.in_panic:
            model.in_panic = True
            model.panic_multiplier = model.panic_multiplier_cfg
    else:
        # Legacy behavior: threshold crossing
        if model.s_panic and not model.in_panic and model.market.S < model.panic_threshold:
            model.in_panic = True
            model.panic_multiplier = model.panic_multiplier_cfg


def scenario_panic_effects(model: "DeFiBankModel") -> None:
    """Cumulative effects when in panic mode:

    * Stablecoin price drops by 0.5% × multiplier per step
    * Liquidity outflow proportionally on each bank
    * Systemic loss accumulated in model.total_loss
    """
    if not model.in_panic:
        return

    drop_frac = 0.005 * model.panic_multiplier

    # 1) Price pressure
    model.market.S *= (1.0 - drop_frac)

    # 2) Liquidity outflow
    for b in model.banks:
        outflow = b.capital * drop_frac
        b.capital -= outflow
        model.total_loss += outflow

    # 3) Clip price to realistic bounds
    model.market.S = np.clip(model.market.S, 0.05, 2.0)


def scenario_cex_intervention(model: "DeFiBankModel") -> None:
    cfg = model.config.get("scenarios", {}).get("cex_intervention", {})
    if not cfg.get("active", False):
        return
    trigger = cfg.get("intervention_trigger", "peg_broken")
    condition_met = (
        (trigger == "peg_broken" and model.market.is_peg_broken())
        or (trigger == "panic_mode" and model.in_panic)
    )
    if not condition_met:
        return
    strength = cfg.get("intervention_strength", 0.25)
    for a in model.schedule.agents:
        if a.__class__.__name__ == "CEXAgent":
            a.intervene(strength)


def scenario_imf_intervention(model: "DeFiBankModel") -> None:
    cfg = model.config.get("scenarios", {}).get("imf_intervention", {})
    if not cfg.get("active", False):
        return
    trigger = cfg.get("intervention_trigger", "panic_mode")
    condition_met = (
        (trigger == "panic_mode" and model.in_panic)
        or (trigger == "peg_broken" and model.market.is_peg_broken())
    )
    if not condition_met:
        return
    strength = cfg.get("intervention_strength", 0.20)
    for a in model.schedule.agents:
        if a.__class__.__name__ == "RegulatorAgent":
            a.intervene(strength)


def scenario_climate_shock(model: "DeFiBankModel") -> None:
    cfg = model.config.get("scenarios", {}).get("climate", {})
    if cfg.get("active") and model.current_step == cfg.get("step"):
        model.market.inject_shock(cfg.get("magnitude", 0.05))


def run_all(model: "DeFiBankModel") -> None:
    # Initialize flags on first step
    if model.current_step == 0:
        load_dynamic_flags(model)

    scenario_depeg(model)
    scenario_panic(model)
    scenario_panic_effects(model)
    scenario_cex_intervention(model)
    scenario_imf_intervention(model)
    scenario_climate_shock(model)
