from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize

from maris_ai.agents.base import PolicyProposal
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint, clamp_speed

def project_joint_slsqp(positions: List[np.ndarray], proposals: List[PolicyProposal], C: ConstraintSpec, max_iters: int, tol: float) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    n = len(proposals)
    vmax = C.v_max - C.margin
    R = C.arena_radius - C.margin
    dmin = C.sep_min + C.margin
    dt = C.step_dt

    v0 = np.concatenate([clamp_speed(p.v_cmd.copy(), vmax) for p in proposals], axis=0)
    vcmd = np.concatenate([p.v_cmd.copy() for p in proposals], axis=0)
    pos = np.concatenate([p.copy() for p in positions], axis=0)

    def obj(v: np.ndarray) -> float:
        dv = v - vcmd
        return float(np.dot(dv, dv))

    cons = []
    for i in range(n):
        def c_speed(v, i=i):
            vi = v[2*i:2*i+2]
            return vmax*vmax - float(np.dot(vi, vi))
        cons.append({"type": "ineq", "fun": c_speed})

    for i in range(n):
        def c_arena(v, i=i):
            vi = v[2*i:2*i+2]
            pi = pos[2*i:2*i+2]
            pn = pi + dt*vi
            return R*R - float(np.dot(pn, pn))
        cons.append({"type": "ineq", "fun": c_arena})

    for i in range(n):
        for j in range(i+1, n):
            def c_sep(v, i=i, j=j):
                vi = v[2*i:2*i+2]
                vj = v[2*j:2*j+2]
                pi = pos[2*i:2*i+2]
                pj = pos[2*j:2*i+2+2]  # deliberate? fix below
                pj = pos[2*j:2*j+2]
                pni = pi + dt*vi
                pnj = pj + dt*vj
                diff = pni - pnj
                return float(np.dot(diff, diff) - dmin*dmin)
            cons.append({"type": "ineq", "fun": c_sep})

    bounds = [(-vmax, vmax)] * (2*n)
    res = minimize(obj, v0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": int(max_iters), "ftol": float(tol), "disp": False})

    v_star = res.x
    actions = [v_star[2*i:2*i+2].copy() for i in range(n)]
    ok, residuals = admissible_joint(positions, actions, C)
    meta: Dict[str, Any] = {
        "projection": {
            "type": "slsqp",
            "success": bool(res.success),
            "status": int(getattr(res, "status", -1)),
            "nit": int(getattr(res, "nit", -1)),
            "fun": float(getattr(res, "fun", float("nan"))),
            "message": str(getattr(res, "message", ""))[:200],
        },
        "ok": bool(ok),
        "residuals": {k: float(v) for k,v in residuals.items()},
    }
    return actions, meta
