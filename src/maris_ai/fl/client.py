from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from maris_ai.agents.base import LinearPolicy

@dataclass
class ClientNode:
    client_id: int
    policy: LinearPolicy

    def local_train(self, rng: np.random.Generator, steps: int, lr: float, hetero: float, adversarial: bool=False) -> LinearPolicy:
        W = self.policy.W.copy()
        b = self.policy.b.copy()
        for _ in range(steps):
            gradW = W + hetero * rng.normal(0, 0.05, size=W.shape)
            gradb = b + hetero * rng.normal(0, 0.05, size=b.shape)
            W = W - lr * gradW
            b = b - lr * gradb
        if adversarial:
            W = W + rng.normal(0, 0.6, size=W.shape)
            b = b + rng.normal(0, 0.6, size=b.shape)
        return LinearPolicy(W=W, b=b)
