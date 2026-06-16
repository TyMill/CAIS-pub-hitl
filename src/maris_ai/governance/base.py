from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from maris_ai.agents.base import PolicyProposal
from maris_ai.governance.constraints import ConstraintSpec

@dataclass(frozen=True)
class GovernanceConfig:
    mode: str
    centralized: bool = True
    projection: str = "heuristic"
    max_iters: int = 64
    tol: float = 1e-6

class GovernanceOperator:
    def apply(self, positions: List, proposals: List[PolicyProposal], C: ConstraintSpec, cfg: GovernanceConfig) -> Tuple[List, Dict[str, Any]]:
        raise NotImplementedError
