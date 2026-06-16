from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
from scipy.stats import mannwhitneyu

def bootstrap_ci(x: np.ndarray, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    boots = []
    for _ in range(int(n_boot)):
        sample = rng.choice(x, size=x.size, replace=True)
        boots.append(float(np.mean(sample)))
    return float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1 - alpha/2))

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (x.size * y.size))

def pairwise_mannwhitney(groups: Dict[str, np.ndarray], alternative: str = "two-sided") -> List[dict]:
    keys = list(groups.keys())
    out = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = np.asarray(groups[keys[i]], dtype=float)
            b = np.asarray(groups[keys[j]], dtype=float)
            stat, p = mannwhitneyu(a, b, alternative=alternative)
            out.append({"group_a": keys[i], "group_b": keys[j], "u_stat": float(stat), "p_value": float(p), "cliffs_delta": cliffs_delta(a, b)})
    return out
