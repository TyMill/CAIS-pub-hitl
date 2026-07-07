#!/usr/bin/env python
"""Aggregate v0.5 revision-pack outputs and generate reviewer-facing tables/plots."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

KEY_COLUMNS = [
    "scenario", "mode", "noise_std", "risk_threshold", "cooldown_steps", "reliability", "delay_steps",
    "violation_rate", "near_miss_rate", "intervention_rate", "risk_mean", "risk_p05", "risk_p25", "risk_p50", "risk_p75", "risk_p95",
    "critical_trigger_rate", "critical_override_count", "suppressed_request_count", "suppressed_critical_count", "cooldown_release_count",
    "repeated_intervention_rate", "post_intervention_violation_rate", "mean_drift_after_intervention", "false_intervention_rate",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("outputs", type=str, help="Root directory produced by scripts/run_revision_pack.py")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    root = Path(args.outputs)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path in root.glob("*/hitl_results.csv"):
        df = pd.read_csv(csv_path)
        df.insert(0, "experiment_group", csv_path.parent.name)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No hitl_results.csv files found under {root}")
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(out / "revision_all_episode_results.csv", index=False)

    summary = data.groupby("experiment_group", as_index=False).mean(numeric_only=True)
    summary.to_csv(out / "revision_group_summary_mean.csv", index=False)

    # compact table for manuscript/rebuttal
    available = ["experiment_group"] + [c for c in KEY_COLUMNS if c in data.columns]
    compact = data[available].groupby("experiment_group", as_index=False).mean(numeric_only=True)
    compact.to_csv(out / "revision_compact_rebuttal_table.csv", index=False)

    # Risk distribution proxy from episode percentiles by group.
    if {"risk_p05", "risk_p25", "risk_p50", "risk_p75", "risk_p95"}.issubset(data.columns):
        risk_cols = ["risk_p05", "risk_p25", "risk_p50", "risk_p75", "risk_p95"]
        dist = data.groupby("experiment_group")[risk_cols].mean().reset_index()
        dist.to_csv(out / "revision_risk_distribution_percentiles.csv", index=False)

    # Figure: threshold vs intervention rate when threshold groups are present.
    threshold_df = data[data["experiment_group"].str.startswith("threshold_")].copy()
    if not threshold_df.empty:
        g = threshold_df.groupby("risk_threshold", as_index=False)[["intervention_rate", "violation_rate", "risk_p50", "risk_p95"]].mean()
        g.to_csv(out / "threshold_response_curve.csv", index=False)
        plt.figure()
        plt.plot(g["risk_threshold"], g["intervention_rate"], marker="o")
        plt.xlabel("Risk activation threshold")
        plt.ylabel("Intervention rate")
        plt.title("Threshold response curve")
        plt.tight_layout()
        plt.savefig(out / "threshold_response_curve.png", dpi=200)
        plt.close()

    # Figure: cooldown diagnostic.
    cool_df = data[data["experiment_group"].str.startswith("cooldown_")].copy()
    if not cool_df.empty:
        g = cool_df.groupby("cooldown_steps", as_index=False)[["intervention_rate", "violation_rate", "suppressed_request_count", "suppressed_critical_count"]].mean()
        g.to_csv(out / "cooldown_diagnostics.csv", index=False)
        plt.figure()
        plt.plot(g["cooldown_steps"], g["intervention_rate"], marker="o")
        plt.xlabel("Cooldown period (simulation steps)")
        plt.ylabel("Intervention rate")
        plt.title("Cooldown intervention reduction")
        plt.tight_layout()
        plt.savefig(out / "cooldown_intervention_rate.png", dpi=200)
        plt.close()

    print({"analysis_dir": str(out), "rows": int(len(data)), "groups": int(data["experiment_group"].nunique())})


if __name__ == "__main__":
    main()
