#!/usr/bin/env python
"""Run the MARIS-AI v0.5 reviewer-response experiment pack.

This script is intentionally conservative: it reuses the original experiment
shape (episodes/steps/seed) and adds only the diagnostics requested during
review: hard-safety override, corrected speed risk, risk distributions, weight
sensitivity, cooldown diagnostics, and stress tests.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from maris_ai.experiments.hitl_runner import sweep_hitl
from maris_ai.governance.constraints import ConstraintSpec
from maris_ai.human.risk import RiskWeights


def _run_group(name: str, args, C: ConstraintSpec, **kwargs) -> dict:
    out_dir = Path(args.outputs) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = sweep_hitl(
        seed=args.seed,
        episodes=args.episodes,
        steps=args.steps,
        n_agents=kwargs.pop("agents", args.agents),
        noise_std=kwargs.pop("noise_std", args.noise_std),
        scenarios=kwargs.pop("scenarios", [args.scenario]),
        modes=kwargs.pop("modes", ["adaptive_hitl"]),
        centralized=True,
        projection=args.projection,
        C=C,
        outputs_dir=out_dir,
        risk_threshold=kwargs.pop("risk_threshold", args.risk_threshold),
        risk_threshold_low=kwargs.pop("risk_threshold_low", args.risk_threshold_low),
        cooldown_steps=kwargs.pop("cooldown_steps", args.cooldown_steps),
        reliability=kwargs.pop("reliability", args.reliability),
        delay_steps=kwargs.pop("delay_steps", args.delay_steps),
        risk_weights=kwargs.pop("risk_weights", args.risk_weights_obj),
        risk_lookahead_steps=kwargs.pop("risk_lookahead_steps", args.risk_lookahead_steps),
        safe_speed_ratio=kwargs.pop("safe_speed_ratio", args.safe_speed_ratio),
        critical_sep_factor=kwargs.pop("critical_sep_factor", args.critical_sep_factor),
        trend_release_delta=kwargs.pop("trend_release_delta", args.trend_release_delta),
    )
    return {"group": name, "outputs": str(out_dir), "summaries": summaries}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--scenario", type=str, default="bottleneck")
    p.add_argument("--noise-std", type=float, default=0.10)
    p.add_argument("--projection", type=str, default="heuristic", choices=["heuristic", "slsqp"])
    p.add_argument("--outputs", type=str, default="outputs_v05_revision_pack")
    p.add_argument("--risk-threshold", type=float, default=0.40)
    p.add_argument("--risk-threshold-low", type=float, default=0.20)
    p.add_argument("--cooldown-steps", type=int, default=5)
    p.add_argument("--reliability", type=float, default=0.95)
    p.add_argument("--delay-steps", type=int, default=0)
    p.add_argument("--risk-weights", type=str, default=None)
    p.add_argument("--risk-lookahead-steps", type=int, default=5)
    p.add_argument("--safe-speed-ratio", type=float, default=0.80)
    p.add_argument("--critical-sep-factor", type=float, default=1.0)
    p.add_argument("--trend-release-delta", type=float, default=0.15)
    p.add_argument("--v-max", type=float, default=2.0)
    p.add_argument("--sep-min", type=float, default=1.0)
    p.add_argument("--arena-radius", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--quick", action="store_true", help="Small smoke run: 3 episodes, 50 steps")
    args = p.parse_args()

    if args.quick:
        args.episodes = 3
        args.steps = 50

    args.risk_weights_obj = RiskWeights.from_string(args.risk_weights)
    C = ConstraintSpec(v_max=args.v_max, sep_min=args.sep_min, arena_radius=args.arena_radius, step_dt=args.dt, margin=args.margin)
    root = Path(args.outputs)
    root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    # 1. Baseline with v0.5 risk semantics for comparability.
    manifest.append(_run_group("baseline_corrected_cram", args, C))

    # 2. Threshold/risk distribution analysis.
    for th in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]:
        manifest.append(_run_group(f"threshold_{th:.2f}", args, C, risk_threshold=th, risk_threshold_low=th / 2))

    # 3. Cooldown diagnostics.
    for cool in [0, 2, 5, 10, 20]:
        manifest.append(_run_group(f"cooldown_{cool}", args, C, cooldown_steps=cool))

    # 4. Weight sensitivity.
    weights = {
        "uniform": RiskWeights(0.25, 0.25, 0.25, 0.25),
        "separation_heavy": RiskWeights(0.60, 0.15, 0.15, 0.10),
        "congestion_heavy": RiskWeights(0.30, 0.40, 0.15, 0.15),
        "speed_heavy": RiskWeights(0.30, 0.20, 0.35, 0.15),
        "uncertainty_heavy": RiskWeights(0.30, 0.20, 0.15, 0.35),
        "baseline": args.risk_weights_obj,
    }
    for name, w in weights.items():
        manifest.append(_run_group(f"weights_{name}", args, C, risk_weights=w))

    # 5. Stress tests requested by reviewers.
    manifest.append(_run_group("stress_high_density", args, C, scenarios=["high_density"], agents=max(args.agents, 12)))
    manifest.append(_run_group("stress_sudden_emergency", args, C, scenarios=["sudden_emergency"]))
    manifest.append(_run_group("stress_boundary", args, C, scenarios=["boundary_stress"]))
    manifest.append(_run_group("stress_noise_spike", args, C, noise_std=max(args.noise_std, 0.30)))

    # 6. Delay/reliability checks kept for consistency with the manuscript.
    for delay in [0, 1, 3, 5, 10]:
        manifest.append(_run_group(f"delay_{delay}", args, C, delay_steps=delay))
    for rel in [1.00, 0.95, 0.90, 0.75, 0.50]:
        manifest.append(_run_group(f"reliability_{rel:.2f}", args, C, reliability=rel))

    payload = {
        "config": {k: (asdict(v) if isinstance(v, RiskWeights) else v) for k, v in vars(args).items() if k != "risk_weights_obj"},
        "effective_risk_weights": asdict(args.risk_weights_obj),
        "groups": manifest,
    }
    (root / "revision_pack_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": str(root), "groups": len(manifest), "manifest": str(root / "revision_pack_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
