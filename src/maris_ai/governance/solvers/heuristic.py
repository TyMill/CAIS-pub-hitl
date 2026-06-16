from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from maris_ai.agents.base import PolicyProposal
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint, clamp_speed

def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def project_joint_heuristic(positions: List[np.ndarray], proposals: List[PolicyProposal], C: ConstraintSpec, max_iters: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    actions = [clamp_speed(p.v_cmd.copy(), C.v_max - C.margin) for p in proposals]
    meta: Dict[str, Any] = {"projection": {"type": "heuristic", "iters": 0, "converged": False}}
    n = len(actions)

    for it in range(max_iters):
        ok, res = admissible_joint(positions, actions, C)
        meta["projection"]["iters"] = it + 1
        if ok:
            meta["projection"]["converged"] = True
            meta["ok"] = True
            meta["residuals"] = {k: float(v) for k,v in res.items()}
            return actions, meta

        dt = C.step_dt
        pos_next = [positions[i] + actions[i] * dt for i in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                vec = pos_next[i] - pos_next[j]
                d = float(np.linalg.norm(vec))
                min_d = C.sep_min + C.margin
                if d < min_d:
                    dir_ = _unit(vec)
                    corr = (min_d - d) / max(dt, 1e-9) * 0.5
                    actions[i] = actions[i] + dir_ * corr
                    actions[j] = actions[j] - dir_ * corr

        R = C.arena_radius - C.margin
        for i in range(n):
            pn = positions[i] + actions[i] * dt
            r = float(np.linalg.norm(pn))
            if r > R:
                to_center = _unit(-pn)
                excess = r - R
                actions[i] = actions[i] + to_center * (excess / max(dt, 1e-9))

        vmax = C.v_max - C.margin
        for i in range(n):
            actions[i] = clamp_speed(actions[i], vmax)

    ok, res = admissible_joint(positions, actions, C)
    meta["ok"] = bool(ok)
    meta["residuals"] = {k: float(v) for k,v in res.items()}
    return actions, meta
