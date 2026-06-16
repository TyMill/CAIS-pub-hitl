from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def make_plots(outputs_dir: str) -> str:
    out_dir = Path(outputs_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create a per-run table
    rows = []
    for run_dir in out_dir.iterdir():
        if not run_dir.is_dir():
            continue
        m = run_dir/"metrics.json"
        meta = run_dir/"meta.json"
        if m.exists() and meta.exists():
            import json as _json
            rows.append({**_json.loads(meta.read_text(encoding="utf-8")), **_json.loads(m.read_text(encoding="utf-8"))})
    if not rows:
        raise FileNotFoundError("No runs found under outputs/. Run sweep first.")

    df = pd.DataFrame(rows)
    df.to_csv(plots_dir/"per_run_metrics.csv", index=False)

    # Plot violin for drift and bar for violation
    plt.figure()
    df.boxplot(column="violation_rate", by="mode")
    plt.title("Violation rate by governance mode")
    plt.suptitle("")
    plt.ylabel("Violation rate")
    plt.tight_layout()
    plt.savefig(plots_dir/"violation_rate_box.png", dpi=200)

    plt.figure()
    df.boxplot(column="mean_drift", by="mode")
    plt.title("Mean drift by governance mode")
    plt.suptitle("")
    plt.ylabel("Mean drift")
    plt.tight_layout()
    plt.savefig(plots_dir/"mean_drift_box.png", dpi=200)

    return str(plots_dir)
