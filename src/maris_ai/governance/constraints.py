from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))

@dataclass(frozen=True)
class ConstraintSpec:
    v_max: float = 2.0
    sep_min: float = 1.0
    arena_radius: float = 20.0
    step_dt: float = 0.2
    margin: float = 0.0

def admissible_joint(positions: List[np.ndarray], actions: List[np.ndarray], C: ConstraintSpec) -> Tuple[bool, Dict[str, float]]:
    vmax = C.v_max - C.margin
    R = C.arena_radius - C.margin
    dt = C.step_dt

    pos_next = []
    max_speed = -1e9
    max_arena = -1e9
    for p,a in zip(positions, actions):
        max_speed = max(max_speed, norm(a) - vmax)
        pn = p + a * dt
        pos_next.append(pn)
        max_arena = max(max_arena, norm(pn) - R)

    max_sep = -1e9
    for i in range(len(pos_next)):
        for j in range(i+1, len(pos_next)):
            d = norm(pos_next[i] - pos_next[j])
            max_sep = max(max_sep, (C.sep_min + C.margin) - d)

    res = {"speed": float(max_speed), "arena": float(max_arena), "separation": float(max_sep)}
    ok = all(v <= 0.0 for v in res.values())
    return ok, res

def clamp_speed(v: np.ndarray, vmax: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= vmax:
        return v
    return v * (vmax / (n + 1e-12))
