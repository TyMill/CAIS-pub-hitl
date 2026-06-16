from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EpisodeMetrics:
    violation_rate: float
    collision_rate: float
    mean_drift: float
    p95_drift: float
    mean_latency_ms: float

def compute_drift(actions, proposals) -> np.ndarray:
    return np.array([float(np.linalg.norm(a - p.v_cmd)) for a,p in zip(actions, proposals)], dtype=float)
