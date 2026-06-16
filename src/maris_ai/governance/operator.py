from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from maris_ai.agents.base import PolicyProposal
from maris_ai.governance.base import GovernanceOperator, GovernanceConfig
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint, clamp_speed
from maris_ai.governance.solvers.heuristic import project_joint_heuristic
from maris_ai.governance.solvers.slsqp import project_joint_slsqp

def fallback_action() -> np.ndarray:
    return np.zeros(2, dtype=float)

class DefaultGovernance(GovernanceOperator):
    def apply(self, positions: List[np.ndarray], proposals: List[PolicyProposal], C: ConstraintSpec, cfg: GovernanceConfig) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if cfg.mode == "none":
            return [p.v_cmd.copy() for p in proposals], {"mode": "none"}

        base = [clamp_speed(p.v_cmd.copy(), C.v_max - C.margin) for p in proposals]

        if cfg.mode == "gate_fallback":
            ok, res = admissible_joint(positions, base, C)
            if ok:
                return base, {"mode": "gate_fallback", "approved": True, "residuals": {k: float(v) for k,v in res.items()}}
            acts = [fallback_action() for _ in base]
            ok2, res2 = admissible_joint(positions, acts, C)
            return acts, {"mode": "gate_fallback", "approved": False, "residuals": {k: float(v) for k,v in res.items()},
                          "fallback_ok": bool(ok2), "fallback_residuals": {k: float(v) for k,v in res2.items()}}

        if cfg.mode == "project":
            if cfg.centralized:
                if cfg.projection == "slsqp":
                    acts, meta = project_joint_slsqp(positions, proposals, C, cfg.max_iters, cfg.tol)
                else:
                    acts, meta = project_joint_heuristic(positions, proposals, C, cfg.max_iters)
                meta["mode"] = "project"
                meta["centralized"] = True
                meta["projection_type"] = cfg.projection
                return acts, meta

            ok, res = admissible_joint(positions, base, C)
            return base, {"mode": "project", "centralized": False, "ok": bool(ok), "residuals": {k: float(v) for k,v in res.items()}}

        raise ValueError(f"Unknown governance mode: {cfg.mode}")
