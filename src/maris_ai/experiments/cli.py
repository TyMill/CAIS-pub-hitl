from __future__ import annotations
import argparse, json
from pathlib import Path
from maris_ai.governance.constraints import ConstraintSpec
from maris_ai.experiments.runner import sweep
from maris_ai.experiments.plots import make_plots
from maris_ai.experiments.report import build_report
from maris_ai.audit.trace import TraceRecord
from maris_ai.audit.replay import verify_hash_chain

def _load_trace(run_dir: Path):
    recs = []
    with (run_dir/"trace.jsonl").open("r", encoding="utf-8") as f:
        import json as _json
        for line in f:
            recs.append(TraceRecord(**_json.loads(line)))
    return recs

def main_sweep():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--scenario", type=str, default="crossing",
                   choices=["head_on","crossing","overtaking","bottleneck","restricted_zone","high_density","sudden_emergency","boundary_stress"])
    p.add_argument("--centralized", action="store_true")
    p.add_argument("--projection", type=str, default="heuristic", choices=["heuristic","slsqp"])
    p.add_argument("--outputs", type=str, default="outputs")
    p.add_argument("--v-max", type=float, default=2.0)
    p.add_argument("--sep-min", type=float, default=1.0)
    p.add_argument("--arena-radius", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.0)
    args = p.parse_args()
    C = ConstraintSpec(v_max=args.v_max, sep_min=args.sep_min, arena_radius=args.arena_radius, step_dt=args.dt, margin=args.margin)
    summaries = sweep(args.seed, args.episodes, args.steps, args.agents, args.noise_std, args.scenario, args.centralized, args.projection, C, Path(args.outputs))
    print(json.dumps(summaries, indent=2))

def main_replay():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    args = p.parse_args()
    recs = _load_trace(Path(args.run_dir))
    ok = verify_hash_chain(recs)
    print(json.dumps({"run_dir": args.run_dir, "records": len(recs), "hash_chain_ok": ok}, indent=2))

def main_plots():
    p = argparse.ArgumentParser()
    p.add_argument("outputs_dir", type=str)
    args = p.parse_args()
    plots_dir = make_plots(args.outputs_dir)
    print(json.dumps({"plots_dir": plots_dir}, indent=2))

def main_report():
    p = argparse.ArgumentParser()
    p.add_argument("outputs_dir", type=str)
    p.add_argument("--metric", type=str, default="violation_rate")
    args = p.parse_args()
    info = build_report(args.outputs_dir, metric=args.metric)
    print(json.dumps(info, indent=2))

def main_federated():
    import numpy as np
    from maris_ai.agents.base import make_policy, observe
    from maris_ai.fl.client import ClientNode
    from maris_ai.fl.server import FederatedServer
    from maris_ai.envs.base import BaseEnv
    from maris_ai.envs.scenarios.generators import crossing, ScenarioParams
    from maris_ai.governance.operator import DefaultGovernance
    from maris_ai.governance.base import GovernanceConfig
    from maris_ai.governance.constraints import admissible_joint, ConstraintSpec

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--bench-states", type=int, default=25)
    p.add_argument("--local-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--hetero", type=float, default=1.0)
    p.add_argument("--adversarial", type=float, default=0.0)
    p.add_argument("--mode", type=str, default="project", choices=["none","gate_fallback","project"])
    p.add_argument("--projection", type=str, default="heuristic", choices=["heuristic","slsqp"])
    p.add_argument("--centralized", action="store_true", default=True)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--outputs", type=str, default="outputs_federated")
    p.add_argument("--v-max", type=float, default=2.0)
    p.add_argument("--sep-min", type=float, default=1.0)
    p.add_argument("--arena-radius", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    obs_dim = 6
    clients = [ClientNode(i, make_policy(rng, obs_dim, 0.6)) for i in range(args.clients)]
    server = FederatedServer(clients)
    fed = server.run_rounds(args.seed, args.rounds, args.local_steps, args.lr, args.hetero, args.adversarial)
    global_policy = fed["global_policy"]

    C = ConstraintSpec(v_max=args.v_max, sep_min=args.sep_min, arena_radius=args.arena_radius, step_dt=args.dt, margin=args.margin)
    G = DefaultGovernance()
    cfg = GovernanceConfig(mode=args.mode, centralized=args.centralized, projection=args.projection)

    bench_rng = np.random.default_rng(args.seed + 999)
    env = BaseEnv(dt=C.step_dt, arena_radius=C.arena_radius)
    violations = 0
    action_samples = []
    drifts = []

    for _ in range(args.bench_states):
        s = env.reset(crossing(int(bench_rng.integers(0, 10_000_000)), ScenarioParams(n_agents=args.agents, arena_radius=C.arena_radius)))
        positions = [ag.pos for ag in s.agents]
        pv = [np.concatenate([ag.pos, ag.vel], axis=0) for ag in s.agents]
        xs = [observe(pv, i, noise_std=args.noise_std, rng=bench_rng) for i in range(args.agents)]
        proposals = [global_policy.propose(x) for x in xs]
        actions, _ = G.apply(positions, proposals, C=C, cfg=cfg)
        ok, _ = admissible_joint(positions, actions, C)
        if not ok:
            violations += 1
        action_samples.append(np.concatenate(actions, axis=0))
        drifts.extend([float(np.linalg.norm(a - p.v_cmd)) for a,p in zip(actions, proposals)])

    A = np.stack(action_samples, axis=0)
    payload = {"config": vars(args), "federated_rounds": fed["round_stats"],
               "bench": {"violations_on_bench": int(violations), "action_var_mean": float(A.var(axis=0).mean()), "mean_drift": float(np.mean(drifts) if drifts else 0.0)}}

    out = Path(args.outputs)
    out.mkdir(parents=True, exist_ok=True)
    path = out/"federated_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(path), "bench": payload["bench"]}, indent=2))

def main_hitl():
    from maris_ai.experiments.hitl_runner import sweep_hitl
    from maris_ai.human.risk import RiskWeights

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--scenarios", type=str, default="head_on,crossing,overtaking,bottleneck,restricted_zone")
    p.add_argument("--modes", type=str, default="none,project,human_approval,human_override,adaptive_hitl")
    p.add_argument("--centralized", action="store_true", default=True)
    p.add_argument("--projection", type=str, default="heuristic", choices=["heuristic", "slsqp"])
    p.add_argument("--outputs", type=str, default="outputs_hitl")
    p.add_argument("--risk-threshold", type=float, default=0.4)
    p.add_argument("--risk-threshold-low", type=float, default=None)
    p.add_argument("--cooldown-steps", type=int, default=5)
    p.add_argument("--reliability", type=float, default=0.95)
    p.add_argument("--delay-steps", type=int, default=0)
    p.add_argument("--risk-weights", type=str, default=None, help="CRAM weights: sep,cong,speed,unc or sep=...,cong=...,speed=...,unc=...")
    p.add_argument("--risk-lookahead-steps", type=int, default=5)
    p.add_argument("--safe-speed-ratio", type=float, default=0.80)
    p.add_argument("--critical-sep-factor", type=float, default=1.0)
    p.add_argument("--trend-release-delta", type=float, default=0.15)
    p.add_argument("--v-max", type=float, default=2.0)
    p.add_argument("--sep-min", type=float, default=1.0)
    p.add_argument("--arena-radius", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.0)
    args = p.parse_args()

    C = ConstraintSpec(
        v_max=args.v_max,
        sep_min=args.sep_min,
        arena_radius=args.arena_radius,
        step_dt=args.dt,
        margin=args.margin,
    )
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    risk_weights = RiskWeights.from_string(args.risk_weights)
    summaries = sweep_hitl(
        seed=args.seed,
        episodes=args.episodes,
        steps=args.steps,
        n_agents=args.agents,
        noise_std=args.noise_std,
        scenarios=scenarios,
        modes=modes,
        centralized=args.centralized,
        projection=args.projection,
        C=C,
        outputs_dir=Path(args.outputs),
        risk_threshold=args.risk_threshold,
        risk_threshold_low=args.risk_threshold_low,
        cooldown_steps=args.cooldown_steps,
        reliability=args.reliability,
        delay_steps=args.delay_steps,
        risk_weights=risk_weights,
        risk_lookahead_steps=args.risk_lookahead_steps,
        safe_speed_ratio=args.safe_speed_ratio,
        critical_sep_factor=args.critical_sep_factor,
        trend_release_delta=args.trend_release_delta,
    )
    print(json.dumps(summaries, indent=2))

