from __future__ import annotations
from typing import List
import numpy as np
from maris_ai.agents.base import LinearPolicy

def fedavg(policies: List[LinearPolicy], weights: np.ndarray) -> LinearPolicy:
    W = sum(w * p.W for w,p in zip(weights, policies))
    b = sum(w * p.b for w,p in zip(weights, policies))
    return LinearPolicy(W=W, b=b)
