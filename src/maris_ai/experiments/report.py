from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from maris_ai.experiments.stats import bootstrap_ci, pairwise_mannwhitney

def _load_episode_metrics(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        m = run_dir / "metrics.json"
        meta = run_dir / "meta.json"
        if m.exists() and meta.exists():
            mj = json.loads(m.read_text(encoding="utf-8"))
            metaj = json.loads(meta.read_text(encoding="utf-8"))
            rows.append({**metaj, **mj, "run_id": run_dir.name})
    if not rows:
        raise FileNotFoundError("No runs found. Run a sweep first to populate outputs/<run_id>/metrics.json")
    return pd.DataFrame(rows)

def build_report(outputs_dir: str, metric: str = "violation_rate") -> Dict[str, Any]:
    out_dir = Path(outputs_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = _load_episode_metrics(out_dir)
    if metric not in df.columns:
        raise ValueError(f"Unknown metric '{metric}'. Available: {sorted(df.columns)}")

    groups = {}
    summary_rows = []
    for mode, g in df.groupby("mode"):
        x = g[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(x, seed=123, n_boot=2000, alpha=0.05)
        summary_rows.append({"mode": mode, "n": int(x.size), "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1) if x.size > 1 else 0.0), "ci95_lo": lo, "ci95_hi": hi})
        groups[mode] = x

    summary = pd.DataFrame(summary_rows).sort_values("mode")
    pairwise = pd.DataFrame(pairwise_mannwhitney(groups, alternative="two-sided"))

    summary_path = reports_dir / "summary_with_ci.csv"
    pairwise_path = reports_dir / "pairwise_tests.csv"
    summary.to_csv(summary_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)

    tex1 = reports_dir / "table_summary.tex"
    tex2 = reports_dir / "table_pairwise.tex"
    summary_fmt = summary.copy()
    for c in ["mean","std","ci95_lo","ci95_hi"]:
        summary_fmt[c] = summary_fmt[c].map(lambda v: f"{v:.4f}")
    summary_fmt.to_latex(tex1, index=False, caption=f"Summary statistics with 95\\% bootstrap CI for {metric}.", label=f"tab:{metric}_summary")

    if not pairwise.empty:
        pairwise_fmt = pairwise.copy()
        for c in ["u_stat","p_value","cliffs_delta"]:
            pairwise_fmt[c] = pairwise_fmt[c].map(lambda v: f"{v:.4g}")
        pairwise_fmt.to_latex(tex2, index=False, caption=f"Pairwise Mann--Whitney U tests for {metric}.", label=f"tab:{metric}_pairwise")

    return {"reports_dir": str(reports_dir), "summary_csv": str(summary_path), "pairwise_csv": str(pairwise_path), "summary_tex": str(tex1), "pairwise_tex": str(tex2), "metric": metric, "n_runs": int(len(df))}
