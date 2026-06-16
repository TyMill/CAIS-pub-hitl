from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class PolicyProposal:
    v_cmd: np.ndarray

@dataclass
class LinearPolicy:
    W: np.ndarray
    b: np.ndarray

    def propose(self, x: np.ndarray) -> PolicyProposal:
        return PolicyProposal(v_cmd=self.W @ x + self.b)

def make_policy(rng: np.random.Generator, obs_dim: int, scale: float = 0.6) -> LinearPolicy:
    return LinearPolicy(W=rng.normal(0, scale, size=(2, obs_dim)), b=rng.normal(0, scale, size=(2,)))

def observe(agents_pos_vel: List[np.ndarray], i: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    own = agents_pos_vel[i]
    own_pos = own[:2]
    own_vel = own[2:]
    rel = np.zeros(2)
    bestd = 1e9
    for j,st in enumerate(agents_pos_vel):
        if j == i:
            continue
        posj = st[:2]
        d = float(np.linalg.norm(posj - own_pos))
        if d < bestd:
            bestd = d
            rel = posj - own_pos
    x = np.concatenate([own_pos, own_vel, rel], axis=0)
    if noise_std > 0:
        x = x + rng.normal(0, noise_std, size=x.shape)
    return x.astype(float)
