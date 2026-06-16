from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import time
import numpy as np

from maris_ai.agents.base import PolicyProposal
from maris_ai.governance.base import GovernanceConfig
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint, clamp_speed
from maris_ai.governance.operator import DefaultGovernance
from maris_ai.human.intervention import choose_human_action
from maris_ai.human.risk import RiskWeights, compute_composite_risk
from maris_ai.human.trigger import AdaptiveTrigger


@dataclass(frozen=True)
class HumanOversightConfig:
    """Configuration for the simulated human oversight layer."""

    mode: str = "adaptive_hitl"
    base_governance_mode: str = "project"
    centralized: bool = True
    projection: str = "heuristic"
    risk_threshold: float = 0.4
    risk_threshold_low: float | None = None
    cooldown_steps: int = 5
    reliability: float = 0.95
    delay_steps: int = 0
    adaptive_only_on_risk: bool = True
    noise_std: float = 0.0
    risk_weights: RiskWeights = RiskWeights()

    @property
    def threshold_high(self) -> float:
        return float(self.risk_threshold)

    @property
    def threshold_low(self) -> float:
        if self.risk_threshold_low is not None:
            return float(self.risk_threshold_low)
        return float(max(0.0, self.risk_threshold * 0.5))


class HumanGovernanceOperator:
    """Human-in-the-loop wrapper around the deterministic CAIS governance operator.

    Version 0.4 introduces risk-aware activation: composite risk assessment,
    hysteresis, and cooldown. The deterministic CAIS operator remains unchanged;
    the human layer decides when to request oversight and whether to accept the
    CAIS action or let a proposal pass under imperfect reliability.
    """

    def __init__(self) -> None:
        self._G = DefaultGovernance()
        self._trigger: AdaptiveTrigger | None = None
        self._trigger_signature: tuple[float, float, int] | None = None

    def _get_trigger(self, cfg: HumanOversightConfig) -> AdaptiveTrigger:
        signature = (cfg.threshold_high, cfg.threshold_low, int(cfg.cooldown_steps))
        if self._trigger is None or self._trigger_signature != signature:
            self._trigger = AdaptiveTrigger(
                threshold_high=cfg.threshold_high,
                threshold_low=cfg.threshold_low,
                cooldown_steps=int(max(0, cfg.cooldown_steps)),
            )
            self._trigger_signature = signature
        return self._trigger

    def apply(
        self,
        positions: list[np.ndarray],
        proposals: list[PolicyProposal],
        C: ConstraintSpec,
        cfg: HumanOversightConfig,
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        start = time.perf_counter()
        base_cfg = GovernanceConfig(
            mode=cfg.base_governance_mode,
            centralized=cfg.centralized,
            projection=cfg.projection,
        )
        cais_actions, cais_meta = self._G.apply(positions, proposals, C=C, cfg=base_cfg)

        proposal_actions = [clamp_speed(p.v_cmd.copy(), C.v_max - C.margin) for p in proposals]
        ok_prop, prop_res = admissible_joint(positions, proposal_actions, C)
        risk = compute_composite_risk(
            positions=positions,
            proposal_actions=proposal_actions,
            proposals=proposals,
            C=C,
            noise_std=cfg.noise_std,
            weights=cfg.risk_weights,
        )
        proposal_risk = float(risk.total)

        trigger_meta: dict[str, Any] = {
            "intervene": False,
            "active": False,
            "cooldown_remaining": 0,
            "reason": "not_adaptive",
        }

        should_intervene = cfg.mode in {"human_approval", "human_override"}
        if cfg.mode == "adaptive_hitl":
            if cfg.adaptive_only_on_risk:
                trigger = self._get_trigger(cfg)
                decision = trigger.evaluate(proposal_risk)
                trigger_meta = decision.to_dict()
                should_intervene = bool(decision.intervene)
            else:
                should_intervene = True
                trigger_meta = {"intervene": True, "active": True, "cooldown_remaining": 0, "reason": "forced_adaptive"}

        if cfg.mode == "cais_only":
            should_intervene = False
            trigger_meta = {"intervene": False, "active": False, "cooldown_remaining": 0, "reason": "cais_only"}

        if should_intervene:
            actions, decision = choose_human_action(
                positions=positions,
                proposal_actions=proposal_actions,
                cais_actions=cais_actions,
                C=C,
                rng=rng,
                mode=cfg.mode,
                reliability=cfg.reliability,
                delay_steps=cfg.delay_steps,
            )
            human_meta: dict[str, Any] = asdict(decision)
        else:
            actions = cais_actions
            human_meta = {
                "intervened": False,
                "reliable": True,
                "success": True,
                "effective_reliability": float(cfg.reliability),
                "reason": "no_intervention_required",
            }

        ok_final, final_res = admissible_joint(positions, actions, C)
        human_meta.update(
            {
                "mode": cfg.mode,
                "proposal_risk": float(proposal_risk),
                "risk": risk.to_dict(),
                "risk_threshold": float(cfg.threshold_high),
                "risk_threshold_high": float(cfg.threshold_high),
                "risk_threshold_low": float(cfg.threshold_low),
                "cooldown_steps": int(cfg.cooldown_steps),
                "trigger": trigger_meta,
                "cooldown_skip": bool(trigger_meta.get("reason") == "cooldown" and not should_intervene),
                "proposal_ok": bool(ok_prop),
                "final_ok": bool(ok_final),
                "proposal_residuals": {k: float(v) for k, v in prop_res.items()},
                "final_residuals": {k: float(v) for k, v in final_res.items()},
                "base_governance": cais_meta,
                "latency_ms": float((time.perf_counter() - start) * 1000.0),
            }
        )
        return actions, human_meta
