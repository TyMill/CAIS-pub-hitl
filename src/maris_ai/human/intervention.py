from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from maris_ai.governance.constraints import ConstraintSpec, admissible_joint
from maris_ai.governance.operator import fallback_action
from maris_ai.human.risk import residual_risk


@dataclass(frozen=True)
class InterventionDecision:
    intervened: bool
    reliable: bool
    success: bool
    effective_reliability: float
    reason: str


def effective_reliability(reliability: float, delay_steps: int, delay_sensitivity: float = 5.0) -> float:
    """Model loss of operator effectiveness due to delayed intervention."""

    reliability = float(np.clip(reliability, 0.0, 1.0))
    if delay_steps <= 0:
        return reliability
    return float(reliability * math.exp(-float(delay_steps) / max(1e-9, delay_sensitivity)))


def choose_human_action(
    positions: list[np.ndarray],
    proposal_actions: list[np.ndarray],
    cais_actions: list[np.ndarray],
    C: ConstraintSpec,
    rng: np.random.Generator,
    mode: str,
    reliability: float,
    delay_steps: int,
) -> tuple[list[np.ndarray], InterventionDecision]:
    """Return action selected by a simulated human oversight layer.

    A reliable operator selects the safer CAIS action. An unreliable operator may
    let the original proposal pass. In override mode, a reliable operator can
    choose a conservative fallback action if the CAIS action is still inadmissible.
    """

    eff = effective_reliability(reliability, delay_steps)
    reliable = bool(rng.random() <= eff)

    ok_cais, _ = admissible_joint(positions, cais_actions, C)
    ok_prop, _ = admissible_joint(positions, proposal_actions, C)

    if mode == "human_approval":
        if reliable:
            return cais_actions, InterventionDecision(True, reliable, bool(ok_cais), eff, "approved_cais_action")
        return proposal_actions, InterventionDecision(True, reliable, bool(ok_prop), eff, "missed_unsafe_proposal")

    if mode == "human_override":
        if reliable:
            if ok_cais:
                return cais_actions, InterventionDecision(True, reliable, True, eff, "override_to_cais_action")
            conservative = [fallback_action() for _ in proposal_actions]
            ok_fallback, _ = admissible_joint(positions, conservative, C)
            return conservative, InterventionDecision(True, reliable, bool(ok_fallback), eff, "override_to_fallback")
        return proposal_actions, InterventionDecision(True, reliable, bool(ok_prop), eff, "failed_override")

    if mode == "adaptive_hitl":
        if reliable:
            return cais_actions, InterventionDecision(True, reliable, bool(ok_cais), eff, "adaptive_human_confirmed")
        return proposal_actions, InterventionDecision(True, reliable, bool(ok_prop), eff, "adaptive_human_missed")

    raise ValueError(f"Unsupported HITL intervention mode: {mode}")
