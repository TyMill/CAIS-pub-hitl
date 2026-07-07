from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np

from maris_ai.agents.base import PolicyProposal
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint


@dataclass(frozen=True)
class RiskWeights:
    """Weights used by the composite HITL risk score.

    The weights are normalized before use. Revision v0.5 keeps the linear
    aggregation as an auditable baseline, but critical hard-safety conditions
    are evaluated separately and can force the total risk to 1.0.
    """

    separation: float = 0.45
    congestion: float = 0.25
    speed: float = 0.15
    uncertainty: float = 0.15

    def normalized(self) -> "RiskWeights":
        total = max(1e-12, self.separation + self.congestion + self.speed + self.uncertainty)
        return RiskWeights(
            separation=self.separation / total,
            congestion=self.congestion / total,
            speed=self.speed / total,
            uncertainty=self.uncertainty / total,
        )

    @staticmethod
    def from_string(payload: str | None) -> "RiskWeights":
        """Parse weights from 'sep,cong,speed,unc' or named 'sep=...,cong=...'."""

        if not payload:
            return RiskWeights()
        text = payload.strip()
        if not text:
            return RiskWeights()
        if "=" not in text:
            parts = [float(x.strip()) for x in text.split(",") if x.strip()]
            if len(parts) != 4:
                raise ValueError("--risk-weights must contain four values: separation,congestion,speed,uncertainty")
            return RiskWeights(*parts)
        values = {"separation": None, "congestion": None, "speed": None, "uncertainty": None}
        aliases = {"sep": "separation", "separation": "separation", "cong": "congestion", "congestion": "congestion", "speed": "speed", "unc": "uncertainty", "uncertainty": "uncertainty"}
        for part in text.split(","):
            if not part.strip():
                continue
            key, value = part.split("=", 1)
            k = aliases.get(key.strip().lower())
            if k is None:
                raise ValueError(f"Unknown risk weight name: {key}")
            values[k] = float(value)
        missing = [k for k, v in values.items() if v is None]
        if missing:
            raise ValueError(f"Missing risk weights: {missing}")
        return RiskWeights(**{k: float(v) for k, v in values.items()})


@dataclass(frozen=True)
class RiskAssessment:
    """Composite risk assessment for risk-aware human governance."""

    total: float
    separation: float
    congestion: float
    speed: float
    uncertainty: float
    residual: float
    critical: bool = False
    critical_separation: bool = False
    critical_boundary: bool = False
    critical_speed: bool = False
    critical_reason: str = "none"
    min_predicted_distance: float = float("inf")
    max_speed_ratio: float = 0.0
    max_boundary_ratio: float = 0.0

    def to_dict(self) -> dict[str, float | bool | str]:
        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, (float, int, np.floating, np.integer)):
                out[k] = float(v)
        return out


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def residual_risk(residuals: dict[str, float], scale: float = 1.0) -> float:
    """Backward-compatible normalized residual risk.

    Positive residuals indicate constraint violations. Instead of returning a raw
    residual with a narrow and poorly calibrated range, this function maps it to
    [0, 1] using a scale parameter. It is kept for compatibility, while the
    preferred method is ``compute_composite_risk``.
    """

    if not residuals:
        return 0.0
    raw = max(0.0, max(float(v) for v in residuals.values()))
    return _clip01(raw / max(1e-12, float(scale)))


def compute_congestion_risk(positions: list[np.ndarray], C: ConstraintSpec, radius_factor: float = 3.0) -> float:
    """Estimate crowding pressure from nearest-neighbour distances."""

    n = len(positions)
    if n < 2:
        return 0.0
    radius = max(C.sep_min * radius_factor, C.sep_min + 1e-12)
    risks: list[float] = []
    for i in range(n):
        nearest = min(_safe_norm(positions[i] - positions[j]) for j in range(n) if j != i)
        risks.append(_clip01((radius - nearest) / max(1e-12, radius - C.sep_min)))
    return float(np.mean(risks))


def compute_speed_risk(
    actions: Iterable[np.ndarray],
    C: ConstraintSpec,
    safe_speed_ratio: float = 0.80,
) -> float:
    """Estimate excessive-speed risk.

    Revision v0.5 replaces the old V/Vmax score. Normal operation below the
    configured safe-speed fraction has zero speed risk; the risk then increases
    only across the remaining margin to the admissible maximum speed.
    """

    vmax = max(1e-12, C.v_max - C.margin)
    v_safe = float(np.clip(safe_speed_ratio, 0.0, 0.999999)) * vmax
    denom = max(1e-12, vmax - v_safe)
    risks = [_clip01((_safe_norm(a) - v_safe) / denom) for a in actions]
    return float(np.mean(risks) if risks else 0.0)


def _minimum_pairwise_distance(points: list[np.ndarray]) -> float:
    if len(points) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            best = min(best, _safe_norm(points[i] - points[j]))
    return float(best)


def compute_separation_risk(
    positions: list[np.ndarray],
    actions: list[np.ndarray],
    C: ConstraintSpec,
    lookahead_steps: int = 5,
) -> tuple[float, float]:
    """Predict finite-horizon separation pressure after the proposed action.

    The model uses a lightweight constant-velocity approximation. The returned
    tuple is (risk, minimum_predicted_distance) over the look-ahead horizon.
    """

    n = len(positions)
    if n < 2:
        return 0.0, float("inf")
    horizon = max(1, int(lookahead_steps))
    danger_radius = max(3.0 * C.sep_min, C.sep_min + 1e-12)
    pair_risks: list[float] = []
    min_distance = float("inf")
    for h in range(1, horizon + 1):
        future_positions = [p + a * C.step_dt * h for p, a in zip(positions, actions)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _safe_norm(future_positions[i] - future_positions[j])
                min_distance = min(min_distance, d)
                pair_risks.append(_clip01((danger_radius - d) / max(1e-12, danger_radius - C.sep_min)))
    if not pair_risks:
        return 0.0, float("inf")
    return float(0.7 * max(pair_risks) + 0.3 * np.mean(pair_risks)), float(min_distance)


def compute_uncertainty_risk(
    proposals: list[PolicyProposal],
    noise_std: float = 0.0,
    reference_noise: float = 0.25,
) -> float:
    """Proxy uncertainty from observation noise and proposal dispersion."""

    if not proposals:
        return 0.0
    proposal_actions = np.stack([p.v_cmd for p in proposals], axis=0)
    dispersion = float(np.mean(np.var(proposal_actions, axis=0)))
    dispersion_risk = _clip01(dispersion / 4.0)
    noise_risk = _clip01(float(noise_std) / max(1e-12, reference_noise))
    return float(0.6 * noise_risk + 0.4 * dispersion_risk)


def compute_critical_conditions(
    positions: list[np.ndarray],
    actions: list[np.ndarray],
    C: ConstraintSpec,
    lookahead_steps: int = 5,
    critical_sep_factor: float = 1.0,
) -> dict[str, float | bool | str]:
    """Evaluate non-compensatory hard-safety conditions.

    These conditions answer the reviewer concern that a linear composite score
    can mask single critical failures. Any positive critical flag forces the
    adaptive trigger to request human oversight or fallback.
    """

    horizon = max(1, int(lookahead_steps))
    vmax = max(1e-12, C.v_max - C.margin)
    arena = max(1e-12, C.arena_radius - C.margin)
    min_dist = float("inf")
    max_boundary = 0.0
    for h in range(1, horizon + 1):
        future = [p + a * C.step_dt * h for p, a in zip(positions, actions)]
        min_dist = min(min_dist, _minimum_pairwise_distance(future))
        max_boundary = max(max_boundary, max((_safe_norm(p) / arena for p in future), default=0.0))
    max_speed_ratio = max((_safe_norm(a) / vmax for a in actions), default=0.0)
    critical_separation = bool(min_dist <= max(1e-12, C.sep_min * float(critical_sep_factor)))
    critical_boundary = bool(max_boundary > 1.0)
    critical_speed = bool(max_speed_ratio > 1.0 + 1e-9)
    reasons = []
    if critical_separation:
        reasons.append("critical_separation")
    if critical_boundary:
        reasons.append("critical_boundary")
    if critical_speed:
        reasons.append("critical_speed")
    return {
        "critical": bool(reasons),
        "critical_separation": critical_separation,
        "critical_boundary": critical_boundary,
        "critical_speed": critical_speed,
        "critical_reason": "+".join(reasons) if reasons else "none",
        "min_predicted_distance": float(min_dist),
        "max_speed_ratio": float(max_speed_ratio),
        "max_boundary_ratio": float(max_boundary),
    }


def compute_composite_risk(
    positions: list[np.ndarray],
    proposal_actions: list[np.ndarray],
    proposals: list[PolicyProposal],
    C: ConstraintSpec,
    noise_std: float = 0.0,
    weights: RiskWeights | None = None,
    lookahead_steps: int = 5,
    safe_speed_ratio: float = 0.80,
    critical_sep_factor: float = 1.0,
) -> RiskAssessment:
    """Compute an auditable [0, 1] risk score for adaptive HITL activation.

    v0.5 uses a two-stage mechanism:
      1. non-compensatory hard-safety override for critical conditions;
      2. interpretable weighted composite risk for non-critical elevated risk.
    """

    w = (weights or RiskWeights()).normalized()
    sep, min_dist = compute_separation_risk(positions, proposal_actions, C, lookahead_steps=lookahead_steps)
    cong = compute_congestion_risk(positions, C)
    speed = compute_speed_risk(proposal_actions, C, safe_speed_ratio=safe_speed_ratio)
    unc = compute_uncertainty_risk(proposals, noise_std=noise_std)
    _, residuals = admissible_joint(positions, proposal_actions, C)
    residual = residual_risk(residuals, scale=max(C.sep_min, C.v_max, 1.0))
    critical = compute_critical_conditions(
        positions,
        proposal_actions,
        C,
        lookahead_steps=lookahead_steps,
        critical_sep_factor=critical_sep_factor,
    )
    total_linear = _clip01(w.separation * sep + w.congestion * cong + w.speed * speed + w.uncertainty * unc)
    total = 1.0 if bool(critical["critical"]) else total_linear
    return RiskAssessment(
        total=total,
        separation=sep,
        congestion=cong,
        speed=speed,
        uncertainty=unc,
        residual=residual,
        critical=bool(critical["critical"]),
        critical_separation=bool(critical["critical_separation"]),
        critical_boundary=bool(critical["critical_boundary"]),
        critical_speed=bool(critical["critical_speed"]),
        critical_reason=str(critical["critical_reason"]),
        min_predicted_distance=float(min(min_dist, float(critical["min_predicted_distance"]))),
        max_speed_ratio=float(critical["max_speed_ratio"]),
        max_boundary_ratio=float(critical["max_boundary_ratio"]),
    )
