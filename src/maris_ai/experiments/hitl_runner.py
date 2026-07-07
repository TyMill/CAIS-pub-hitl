from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from maris_ai.agents.base import make_policy, observe
from maris_ai.audit.hashing import stable_hash
from maris_ai.audit.replay import verify_hash_chain
from maris_ai.audit.trace import TraceRecord, make_trace_record
from maris_ai.envs.base import BaseEnv
from maris_ai.envs.scenarios.generators import SCENARIOS, ScenarioParams
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint
from maris_ai.governance.base import GovernanceConfig
from maris_ai.governance.operator import DefaultGovernance
from maris_ai.human.metrics import HITLMetrics
from maris_ai.human.operator import HumanGovernanceOperator, HumanOversightConfig
from maris_ai.human.risk import RiskWeights


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _collision_count(positions: list[np.ndarray], sep_min: float) -> int:
    c = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if float(np.linalg.norm(positions[i] - positions[j])) < (sep_min * 0.5):
                c += 1
    return c


def _near_miss_count(positions: list[np.ndarray], sep_min: float, factor: float = 1.5) -> int:
    c = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if sep_min <= d < sep_min * factor:
                c += 1
    return c


def _drifts(actions, proposals) -> np.ndarray:
    return np.array([float(np.linalg.norm(a - p.v_cmd)) for a, p in zip(actions, proposals)], dtype=float)


def _apply_mode(
    positions: list[np.ndarray],
    proposals,
    C: ConstraintSpec,
    mode: str,
    centralized: bool,
    projection: str,
    risk_threshold: float,
    reliability: float,
    delay_steps: int,
    rng: np.random.Generator,
    human_operator: HumanGovernanceOperator | None = None,
    risk_threshold_low: float | None = None,
    cooldown_steps: int = 5,
    noise_std: float = 0.0,
    risk_weights: RiskWeights | None = None,
    risk_lookahead_steps: int = 5,
    safe_speed_ratio: float = 0.80,
    critical_sep_factor: float = 1.0,
    trend_release_delta: float = 0.15,
):
    if mode in {"none", "gate_fallback", "project"}:
        G = DefaultGovernance()
        cfg = GovernanceConfig(mode=mode, centralized=centralized, projection=projection)
        start = time.perf_counter()
        actions, meta = G.apply(positions, proposals, C=C, cfg=cfg)
        meta = dict(meta)
        meta.update(
            {
                "hitl_mode": mode,
                "human_intervened": False,
                "human_success": True,
                "human_reliable": True,
                "proposal_risk": 0.0,
                "risk": {},
                "cooldown_skip": False,
                "trigger": {"active": False, "reason": "non_hitl_mode"},
                "latency_ms": float((time.perf_counter() - start) * 1000.0),
                "risk_weights": {},
            }
        )
        return actions, meta

    H = human_operator or HumanGovernanceOperator()
    cfg = HumanOversightConfig(
        mode=mode,
        base_governance_mode="project",
        centralized=centralized,
        projection=projection,
        risk_threshold=risk_threshold,
        risk_threshold_low=risk_threshold_low,
        cooldown_steps=cooldown_steps,
        reliability=reliability,
        delay_steps=delay_steps,
        noise_std=noise_std,
        risk_weights=risk_weights or RiskWeights(),
        risk_lookahead_steps=risk_lookahead_steps,
        safe_speed_ratio=safe_speed_ratio,
        critical_sep_factor=critical_sep_factor,
        trend_release_delta=trend_release_delta,
    )
    actions, meta = H.apply(positions, proposals, C=C, cfg=cfg, rng=rng)
    meta["hitl_mode"] = mode
    meta["human_intervened"] = bool(meta.get("intervened", False))
    meta["human_success"] = bool(meta.get("success", True))
    meta["human_reliable"] = bool(meta.get("reliable", True))
    return actions, meta


def run_hitl_episode(
    seed: int,
    scenario: str,
    mode: str,
    centralized: bool,
    projection: str,
    steps: int,
    n_agents: int,
    noise_std: float,
    C: ConstraintSpec,
    out_dir: Path,
    risk_threshold: float = 0.4,
    reliability: float = 0.95,
    delay_steps: int = 0,
    risk_threshold_low: float | None = None,
    cooldown_steps: int = 5,
    risk_weights: RiskWeights | None = None,
    risk_lookahead_steps: int = 5,
    safe_speed_ratio: float = 0.80,
    critical_sep_factor: float = 1.0,
    trend_release_delta: float = 0.15,
) -> HITLMetrics:
    rng = np.random.default_rng(seed)
    init = SCENARIOS[scenario](seed, ScenarioParams(n_agents=n_agents, arena_radius=C.arena_radius))
    env = BaseEnv(dt=C.step_dt, arena_radius=C.arena_radius)
    s = env.reset(init)

    obs_dim = 6
    policies = [make_policy(rng, obs_dim=obs_dim, scale=0.6) for _ in range(n_agents)]
    policy_ids = [stable_hash({"W": p.W.tolist(), "b": p.b.tolist()}) for p in policies]
    human_operator = HumanGovernanceOperator()

    records: list[TraceRecord] = []
    prev_hash = "GENESIS"
    violations = 0
    collisions = 0
    near_misses = 0
    interventions = 0
    successful_overrides = 0
    drifts = []
    latencies = []
    risks = []
    separation_risks = []
    congestion_risks = []
    speed_risks = []
    uncertainty_risks = []
    confidence_values = []
    cooldown_skips = 0
    trigger_active_steps = 0
    false_interventions = 0
    suppressed_critical_count = 0
    cooldown_release_count = 0
    critical_override_count = 0
    critical_steps = 0
    repeated_interventions = 0
    previous_intervention_step: int | None = None
    last_intervention_step: int | None = None
    post_intervention_window_steps = 0
    post_intervention_violations = 0
    drift_after_intervention: list[float] = []
    min_predicted_distances: list[float] = []

    for t in range(steps):
        positions = [ag.pos for ag in s.agents]
        velocities = [ag.vel for ag in s.agents]
        pv = [np.concatenate([ag.pos, ag.vel], axis=0) for ag in s.agents]
        xs = [observe(pv, i, noise_std=noise_std, rng=rng) for i in range(n_agents)]
        proposals = [policies[i].propose(xs[i]) for i in range(n_agents)]

        actions, meta = _apply_mode(
            positions=positions,
            proposals=proposals,
            C=C,
            mode=mode,
            centralized=centralized,
            projection=projection,
            risk_threshold=risk_threshold,
            reliability=reliability,
            delay_steps=delay_steps,
            rng=rng,
            human_operator=human_operator,
            risk_threshold_low=risk_threshold_low,
            cooldown_steps=cooldown_steps,
            noise_std=noise_std,
            risk_weights=risk_weights,
            risk_lookahead_steps=risk_lookahead_steps,
            safe_speed_ratio=safe_speed_ratio,
            critical_sep_factor=critical_sep_factor,
            trend_release_delta=trend_release_delta,
        )
        latencies.append(float(meta.get("latency_ms", 0.0)))
        risk_payload = meta.get("risk", {}) if isinstance(meta.get("risk", {}), dict) else {}
        risks.append(float(meta.get("proposal_risk", risk_payload.get("total", 0.0)) or 0.0))
        separation_risks.append(float(risk_payload.get("separation", 0.0)))
        congestion_risks.append(float(risk_payload.get("congestion", 0.0)))
        speed_risks.append(float(risk_payload.get("speed", 0.0)))
        uncertainty_risks.append(float(risk_payload.get("uncertainty", 0.0)))
        min_predicted_distances.append(float(risk_payload.get("min_predicted_distance", float("nan"))))
        is_critical = bool(risk_payload.get("critical", False))
        if is_critical:
            critical_steps += 1
        confidence_values.append(float(meta.get("effective_reliability", reliability)))
        if bool(meta.get("cooldown_skip", False)):
            cooldown_skips += 1
            if is_critical:
                suppressed_critical_count += 1
        trigger_meta = meta.get("trigger", {}) if isinstance(meta.get("trigger", {}), dict) else {}
        if bool(trigger_meta.get("cooldown_released", False)):
            cooldown_release_count += 1
        if trigger_meta.get("reason") == "critical_override":
            critical_override_count += 1
        if bool(trigger_meta.get("active", False)):
            trigger_active_steps += 1

        ok, res = admissible_joint(positions, actions, C)
        if not ok:
            violations += 1
        if last_intervention_step is not None and 1 <= (t - last_intervention_step) <= 3:
            post_intervention_window_steps += 1
            if not ok:
                post_intervention_violations += 1
        collisions += _collision_count(positions, C.sep_min)
        near_misses += _near_miss_count(positions, C.sep_min)

        step_drifts = _drifts(actions, proposals).tolist()
        if bool(meta.get("human_intervened", False)):
            interventions += 1
            if previous_intervention_step is not None and t - previous_intervention_step == 1:
                repeated_interventions += 1
            previous_intervention_step = t
            last_intervention_step = t
            drift_after_intervention.extend(step_drifts)
            if bool(meta.get("human_success", False)) and ok:
                successful_overrides += 1
            if bool(meta.get("proposal_ok", False)) and bool(meta.get("final_ok", False)):
                false_interventions += 1

        drifts.extend(step_drifts)

        rec = make_trace_record(
            t=t,
            scenario=s.scenario,
            positions=[p.tolist() for p in positions],
            velocities=[v.tolist() for v in velocities],
            proposals=proposals,
            actions=actions,
            residuals={k: float(v) for k, v in res.items()},
            governance_meta=meta,
            seed=seed,
            policy_ids=policy_ids,
            C=asdict(C),
            prev_hash=prev_hash,
        )
        prev_hash = rec.this_hash
        records.append(rec)
        s = env.step(s, actions)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=False, default=_json_default) + "\n")

    meta_payload = {
        "seed": seed,
        "scenario": scenario,
        "mode": mode,
        "centralized": centralized,
        "projection": projection,
        "steps": steps,
        "n_agents": n_agents,
        "noise_std": noise_std,
        "risk_threshold": risk_threshold,
        "risk_threshold_low": risk_threshold_low,
        "cooldown_steps": cooldown_steps,
        "reliability": reliability,
        "delay_steps": delay_steps,
        "risk_weights": asdict(risk_weights or RiskWeights()),
        "risk_lookahead_steps": risk_lookahead_steps,
        "safe_speed_ratio": safe_speed_ratio,
        "critical_sep_factor": critical_sep_factor,
        "trend_release_delta": trend_release_delta,
        "C": asdict(C),
        "policy_ids": policy_ids,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    (out_dir / "trace_ok.json").write_text(json.dumps({"hash_chain_ok": verify_hash_chain(records)}, indent=2), encoding="utf-8")

    drifts_np = np.array(drifts, dtype=float)
    risks_np = np.array(risks, dtype=float)
    finite_min_dist = np.array([x for x in min_predicted_distances if np.isfinite(x)], dtype=float)
    drift_after_np = np.array(drift_after_intervention, dtype=float)
    em = HITLMetrics(
        violation_rate=float(violations / max(1, steps)),
        collision_rate=float(collisions / max(1, steps)),
        near_miss_rate=float(near_misses / max(1, steps)),
        intervention_rate=float(interventions / max(1, steps)),
        override_success_rate=float(successful_overrides / max(1, interventions)),
        mean_intervention_delay=float(delay_steps if interventions > 0 else 0.0),
        operator_workload=float(interventions / max(1, steps)),
        mean_drift=float(drifts_np.mean() if drifts_np.size else 0.0),
        p95_drift=float(np.quantile(drifts_np, 0.95) if drifts_np.size else 0.0),
        mean_latency_ms=float(np.mean(latencies) if latencies else 0.0),
        risk_mean=float(np.mean(risks_np) if risks_np.size else 0.0),
        risk_p05=float(np.quantile(risks_np, 0.05) if risks_np.size else 0.0),
        risk_p25=float(np.quantile(risks_np, 0.25) if risks_np.size else 0.0),
        risk_p50=float(np.quantile(risks_np, 0.50) if risks_np.size else 0.0),
        risk_p75=float(np.quantile(risks_np, 0.75) if risks_np.size else 0.0),
        risk_p95=float(np.quantile(risks_np, 0.95) if risks_np.size else 0.0),
        separation_risk_mean=float(np.mean(separation_risks) if separation_risks else 0.0),
        congestion_risk_mean=float(np.mean(congestion_risks) if congestion_risks else 0.0),
        speed_risk_mean=float(np.mean(speed_risks) if speed_risks else 0.0),
        uncertainty_risk_mean=float(np.mean(uncertainty_risks) if uncertainty_risks else 0.0),
        human_activation_count=int(interventions),
        cooldown_skips=int(cooldown_skips),
        suppressed_request_count=int(cooldown_skips),
        suppressed_critical_count=int(suppressed_critical_count),
        cooldown_release_count=int(cooldown_release_count),
        critical_override_count=int(critical_override_count),
        critical_trigger_rate=float(critical_steps / max(1, steps)),
        trigger_active_rate=float(trigger_active_steps / max(1, steps)),
        confidence_mean=float(np.mean(confidence_values) if confidence_values else 0.0),
        false_intervention_rate=float(false_interventions / max(1, interventions)),
        repeated_intervention_rate=float(repeated_interventions / max(1, interventions)),
        post_intervention_violation_rate=float(post_intervention_violations / max(1, post_intervention_window_steps)),
        mean_drift_after_intervention=float(np.mean(drift_after_np) if drift_after_np.size else 0.0),
        min_predicted_distance_mean=float(np.mean(finite_min_dist) if finite_min_dist.size else 0.0),
    )
    (out_dir / "metrics.json").write_text(json.dumps(em.__dict__, indent=2), encoding="utf-8")
    return em


def sweep_hitl(
    seed: int,
    episodes: int,
    steps: int,
    n_agents: int,
    noise_std: float,
    scenarios: list[str],
    modes: list[str],
    centralized: bool,
    projection: str,
    C: ConstraintSpec,
    outputs_dir: Path,
    risk_threshold: float = 0.4,
    reliability: float = 0.95,
    delay_steps: int = 0,
    risk_threshold_low: float | None = None,
    cooldown_steps: int = 5,
    risk_weights: RiskWeights | None = None,
    risk_lookahead_steps: int = 5,
    safe_speed_ratio: float = 0.80,
    critical_sep_factor: float = 1.0,
    trend_release_delta: float = 0.15,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for mode in modes:
            for ep in range(episodes):
                ep_seed = seed + ep * 1000 + 17
                run_id = (
                    f"hitl_{scenario}_{mode}_noise{noise_std:g}_thr{risk_threshold:g}_"
                    f"low{risk_threshold_low if risk_threshold_low is not None else 'auto'}_"
                    f"cool{cooldown_steps}_rel{reliability:g}_delay{delay_steps}_seed{ep_seed}"
                )
                metrics = run_hitl_episode(
                    seed=ep_seed,
                    scenario=scenario,
                    mode=mode,
                    centralized=centralized,
                    projection=projection,
                    steps=steps,
                    n_agents=n_agents,
                    noise_std=noise_std,
                    C=C,
                    out_dir=outputs_dir / run_id,
                    risk_threshold=risk_threshold,
                    reliability=reliability,
                    delay_steps=delay_steps,
                    risk_threshold_low=risk_threshold_low,
                    cooldown_steps=cooldown_steps,
                    risk_weights=risk_weights,
                    risk_lookahead_steps=risk_lookahead_steps,
                    safe_speed_ratio=safe_speed_ratio,
                    critical_sep_factor=critical_sep_factor,
                    trend_release_delta=trend_release_delta,
                )
                row = {
                    "run_id": run_id,
                    "scenario": scenario,
                    "mode": mode,
                    "seed": ep_seed,
                    "noise_std": noise_std,
                    "risk_threshold": risk_threshold,
                    "risk_threshold_low": risk_threshold_low if risk_threshold_low is not None else risk_threshold * 0.5,
                    "cooldown_steps": cooldown_steps,
                    "reliability": reliability,
                    "delay_steps": delay_steps,
                    "risk_weights": json.dumps(asdict(risk_weights or RiskWeights())),
                    "risk_lookahead_steps": risk_lookahead_steps,
                    "safe_speed_ratio": safe_speed_ratio,
                    "critical_sep_factor": critical_sep_factor,
                    "trend_release_delta": trend_release_delta,
                    **metrics.__dict__,
                }
                rows.append(row)
            summaries.append(
                {
                    "scenario": scenario,
                    "mode": mode,
                    "episodes": int(episodes),
                    "noise_std": float(noise_std),
                    "risk_threshold": float(risk_threshold),
                    "risk_threshold_low": float(risk_threshold_low if risk_threshold_low is not None else risk_threshold * 0.5),
                    "cooldown_steps": int(cooldown_steps),
                    "reliability": float(reliability),
                    "delay_steps": int(delay_steps),
                    "risk_weights": asdict(risk_weights or RiskWeights()),
                    "risk_lookahead_steps": int(risk_lookahead_steps),
                    "safe_speed_ratio": float(safe_speed_ratio),
                    "critical_sep_factor": float(critical_sep_factor),
                    "trend_release_delta": float(trend_release_delta),
                }
            )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "hitl_sweep_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(outputs_dir / "hitl_results.csv", index=False)
    except Exception:
        (outputs_dir / "hitl_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return summaries
