from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from maris_ai.fl.client import ClientNode
from maris_ai.fl.aggregation import fedavg

@dataclass
class FederatedServer:
    clients: List[ClientNode]

    def run_rounds(self, seed: int, rounds: int, local_steps: int, lr: float, hetero: float, adversarial_fraction: float = 0.0) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        n = len(self.clients)
        weights = np.ones(n, dtype=float) / n
        global_policy = fedavg([c.policy for c in self.clients], weights)

        stats = []
        for r in range(rounds):
            updated = []
            for i,c in enumerate(self.clients):
                adv = adversarial_fraction > 0 and i < int(n * adversarial_fraction)
                h = hetero * (1.0 + (i % 3) * 0.3)
                updated.append(c.local_train(rng, steps=local_steps, lr=lr, hetero=h, adversarial=adv))
            global_policy = fedavg(updated, weights)
            for c in self.clients:
                c.policy = global_policy
            stats.append({"round": int(r), "global_norm": float(np.linalg.norm(global_policy.W))})
        return {"round_stats": stats, "global_policy": global_policy}
