from __future__ import annotations
import json, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from maris_ai.envs.base import BaseEnv
from maris_ai.envs.scenarios.generators import SCENARIOS, ScenarioParams
from maris_ai.agents.base import make_policy, observe
from maris_ai.governance.constraints import ConstraintSpec, admissible_joint
from maris_ai.governance.base import GovernanceConfig
from maris_ai.governance.operator import DefaultGovernance
from maris_ai.audit.trace import make_trace_record, TraceRecord
from maris_ai.audit.replay import verify_hash_chain
from maris_ai.audit.hashing import stable_hash
from maris_ai.experiments.metrics import EpisodeMetrics, compute_drift

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _collision_count(positions: List[np.ndarray], sep_min: float) -> int:
    c = 0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if float(np.linalg.norm(positions[i] - positions[j])) < (sep_min * 0.5):
                c += 1
    return c

def run_episode(seed: int, scenario: str, mode: str, centralized: bool, projection: str, steps: int,
                n_agents: int, noise_std: float, C: ConstraintSpec, out_dir: Path) -> EpisodeMetrics:
    rng = np.random.default_rng(seed)
    init = SCENARIOS[scenario](seed, ScenarioParams(n_agents=n_agents, arena_radius=C.arena_radius))
    env = BaseEnv(dt=C.step_dt, arena_radius=C.arena_radius)
    s = env.reset(init)

    obs_dim = 6
    policies = [make_policy(rng, obs_dim=obs_dim, scale=0.6) for _ in range(n_agents)]
    policy_ids = [stable_hash({"W": p.W.tolist(), "b": p.b.tolist()}) for p in policies]

    G = DefaultGovernance()
    cfg = GovernanceConfig(mode=mode, centralized=centralized, projection=projection)

    records: List[TraceRecord] = []
    prev_hash = "GENESIS"
    violations = 0
    collisions = 0
    drifts = []
    latencies = []

    for t in range(steps):
        positions = [ag.pos for ag in s.agents]
        velocities = [ag.vel for ag in s.agents]
        pv = [np.concatenate([ag.pos, ag.vel], axis=0) for ag in s.agents]
        xs = [observe(pv, i, noise_std=noise_std, rng=rng) for i in range(n_agents)]
        proposals = [policies[i].propose(xs[i]) for i in range(n_agents)]

        start = time.perf_counter()
        actions, meta = G.apply(positions, proposals, C=C, cfg=cfg)
        latencies.append((time.perf_counter() - start) * 1000.0)

        ok, res = admissible_joint(positions, actions, C)
        if not ok:
            violations += 1
        collisions += _collision_count(positions, C.sep_min)

        dr = compute_drift(actions, proposals)
        drifts.extend(dr.tolist())

        rec = make_trace_record(
            t=t, scenario=s.scenario,
            positions=[p.tolist() for p in positions],
            velocities=[v.tolist() for v in velocities],
            proposals=proposals, actions=actions,
            residuals={k: float(v) for k,v in res.items()},
            governance_meta=meta, seed=seed,
            policy_ids=policy_ids, C=asdict(C),
            prev_hash=prev_hash,
        )
        prev_hash = rec.this_hash
        records.append(rec)
        s = env.step(s, actions)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/"trace.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=False, default=_json_default) + "\n")

    (out_dir/"meta.json").write_text(json.dumps({
        "seed": seed, "scenario": scenario, "mode": mode, "centralized": centralized, "projection": projection,
        "steps": steps, "n_agents": n_agents, "noise_std": noise_std, "C": asdict(C), "policy_ids": policy_ids,
    }, indent=2), encoding="utf-8")

    (out_dir/"trace_ok.json").write_text(json.dumps({"hash_chain_ok": verify_hash_chain(records)}, indent=2), encoding="utf-8")

    drifts_np = np.array(drifts, dtype=float)
    em = EpisodeMetrics(
        violation_rate=float(violations / max(1, steps)),
        collision_rate=float(collisions / max(1, steps)),
        mean_drift=float(drifts_np.mean() if drifts_np.size else 0.0),
        p95_drift=float(np.quantile(drifts_np, 0.95) if drifts_np.size else 0.0),
        mean_latency_ms=float(float(np.mean(latencies)) if latencies else 0.0),
    )
    (out_dir/"metrics.json").write_text(json.dumps(em.__dict__, indent=2), encoding="utf-8")
    return em

def sweep(seed: int, episodes: int, steps: int, n_agents: int, noise_std: float, scenario: str,
          centralized: bool, projection: str, C: ConstraintSpec, outputs_dir: Path) -> List[Dict[str, Any]]:
    summaries = []
    for mode in ["none", "gate_fallback", "project"]:
        for ep in range(episodes):
            ep_seed = seed + ep * 1000 + 17
            run_id = f"{scenario}_{mode}_cent{int(centralized)}_{projection}_seed{ep_seed}"
            run_episode(ep_seed, scenario, mode, centralized, projection, steps, n_agents, noise_std, C, outputs_dir / run_id)
        summaries.append({"scenario": scenario, "mode": mode, "centralized": bool(centralized), "projection": projection, "episodes": int(episodes)})
    (outputs_dir/"sweep_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
