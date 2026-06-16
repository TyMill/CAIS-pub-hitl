from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np

from maris_ai.agents.base import PolicyProposal
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint


@dataclass(frozen=True)
class RiskWeights:
    """Weights used by the composite HITL risk score."""

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


@dataclass(frozen=True)
class RiskAssessment:
    """Composite risk assessment for risk-aware human governance."""

    total: float
    separation: float
    congestion: float
    speed: float
    uncertainty: float
    residual: float

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


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
    """Estimate crowding pressure from nearest-neighbour distances.

    The score approaches 1 when agents are closer than ``sep_min`` and approaches
    0 when all pairwise distances exceed ``radius_factor * sep_min``.
    """

    n = len(positions)
    if n < 2:
        return 0.0
    radius = max(C.sep_min * radius_factor, C.sep_min + 1e-12)
    risks: list[float] = []
    for i in range(n):
        nearest = min(_safe_norm(positions[i] - positions[j]) for j in range(n) if j != i)
        risks.append(_clip01((radius - nearest) / max(1e-12, radius - C.sep_min)))
    return float(np.mean(risks))


def compute_speed_risk(actions: Iterable[np.ndarray], C: ConstraintSpec) -> float:
    """Estimate how close proposed actions are to the speed boundary."""

    vmax = max(1e-12, C.v_max - C.margin)
    ratios = [_clip01(_safe_norm(a) / vmax) for a in actions]
    if not ratios:
        return 0.0
    # Nonlinear scaling prevents medium-speed nominal actions from triggering HITL too often.
    return float(np.mean(np.square(ratios)))


def compute_separation_risk(positions: list[np.ndarray], actions: list[np.ndarray], C: ConstraintSpec) -> float:
    """Predict next-step separation pressure after the proposed action."""

    n = len(positions)
    if n < 2:
        return 0.0
    next_positions = [p + a * C.step_dt for p, a in zip(positions, actions)]
    danger_radius = max(3.0 * C.sep_min, C.sep_min + 1e-12)
    pair_risks: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = _safe_norm(next_positions[i] - next_positions[j])
            pair_risks.append(_clip01((danger_radius - d) / max(1e-12, danger_radius - C.sep_min)))
    if not pair_risks:
        return 0.0
    # Max emphasizes the most dangerous pair, mean prevents all scenarios from saturating.
    return float(0.7 * max(pair_risks) + 0.3 * np.mean(pair_risks))


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


def compute_composite_risk(
    positions: list[np.ndarray],
    proposal_actions: list[np.ndarray],
    proposals: list[PolicyProposal],
    C: ConstraintSpec,
    noise_std: float = 0.0,
    weights: RiskWeights | None = None,
) -> RiskAssessment:
    """Compute a calibrated [0, 1] risk score for adaptive HITL activation.

    The score combines predicted separation pressure, congestion, speed-boundary
    pressure, and uncertainty. This replaces the previous raw max-residual trigger,
    which behaved almost binarily and made threshold sweeps uninformative.
    """

    w = (weights or RiskWeights()).normalized()
    sep = compute_separation_risk(positions, proposal_actions, C)
    cong = compute_congestion_risk(positions, C)
    speed = compute_speed_risk(proposal_actions, C)
    unc = compute_uncertainty_risk(proposals, noise_std=noise_std)
    _, residuals = admissible_joint(positions, proposal_actions, C)
    residual = residual_risk(residuals, scale=max(C.sep_min, C.v_max, 1.0))
    total = _clip01(w.separation * sep + w.congestion * cong + w.speed * speed + w.uncertainty * unc)
    return RiskAssessment(total=total, separation=sep, congestion=cong, speed=speed, uncertainty=unc, residual=residual)
