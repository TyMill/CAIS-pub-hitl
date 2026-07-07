from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HITLMetrics:
    """Episode-level metrics for human-in-the-loop CAIS experiments."""

    violation_rate: float
    collision_rate: float
    near_miss_rate: float
    intervention_rate: float
    override_success_rate: float
    mean_intervention_delay: float
    operator_workload: float
    mean_drift: float
    p95_drift: float
    mean_latency_ms: float
    risk_mean: float = 0.0
    risk_p05: float = 0.0
    risk_p25: float = 0.0
    risk_p50: float = 0.0
    risk_p75: float = 0.0
    risk_p95: float = 0.0
    separation_risk_mean: float = 0.0
    congestion_risk_mean: float = 0.0
    speed_risk_mean: float = 0.0
    uncertainty_risk_mean: float = 0.0
    human_activation_count: int = 0
    cooldown_skips: int = 0
    suppressed_request_count: int = 0
    suppressed_critical_count: int = 0
    cooldown_release_count: int = 0
    critical_override_count: int = 0
    critical_trigger_rate: float = 0.0
    trigger_active_rate: float = 0.0
    confidence_mean: float = 0.0
    false_intervention_rate: float = 0.0
    repeated_intervention_rate: float = 0.0
    post_intervention_violation_rate: float = 0.0
    mean_drift_after_intervention: float = 0.0
    min_predicted_distance_mean: float = 0.0
