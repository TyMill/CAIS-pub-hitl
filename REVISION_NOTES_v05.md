# MARIS-AI v0.5 revision notes for AHO manuscript

This branch implements the reviewer-response experiment pack discussed for the Adaptive Human Oversight manuscript.

## Implemented model changes

1. **Two-stage CRAM**
   - Critical hard-safety conditions are checked before the weighted composite score.
   - Critical separation, boundary, or speed conditions force `risk.total = 1.0`.
   - Critical conditions bypass cooldown via trigger reason `critical_override`.

2. **Corrected speed risk**
   - Replaced naive `V / Vmax` behavior with excessive-speed risk:
     - zero risk below `safe_speed_ratio * Vmax`,
     - increasing risk only in the excessive-speed margin.

3. **Finite look-ahead separation risk**
   - Separation risk now estimates minimum predicted distance over configurable future decision cycles.
   - Default: `--risk-lookahead-steps 5`.

4. **Trend-aware cooldown**
   - Cooldown suppresses repeated requests.
   - It releases early if risk rises by `--trend-release-delta` or if a critical condition appears.

5. **Human intervention diagnostics**
   - Added counts and rates needed for rebuttal:
     - `suppressed_request_count`,
     - `suppressed_critical_count`,
     - `critical_override_count`,
     - `cooldown_release_count`,
     - `repeated_intervention_rate`,
     - `post_intervention_violation_rate`,
     - `mean_drift_after_intervention`,
     - risk percentiles.

6. **Stress-test scenarios**
   - `high_density`,
   - `sudden_emergency`,
   - `boundary_stress`,
   - plus noise spike via `--noise-std 0.30`.

## How to run

Small smoke test:

```bash
pip install -e .
python scripts/run_revision_pack.py --quick --outputs outputs_v05_smoke
python scripts/analyze_revision_outputs.py outputs_v05_smoke
```

Full reviewer-response pack:

```bash
python scripts/run_revision_pack.py --episodes 50 --steps 150 --outputs outputs_v05_revision_pack
python scripts/analyze_revision_outputs.py outputs_v05_revision_pack
```

## Reviewer-facing outputs

After analysis, use:

- `analysis/revision_compact_rebuttal_table.csv` for the response letter,
- `analysis/revision_risk_distribution_percentiles.csv` for the threshold plateau argument,
- `analysis/cooldown_diagnostics.csv` for the cooldown explanation,
- `analysis/threshold_response_curve.png` and `analysis/cooldown_intervention_rate.png` as optional figures.

## Interpretation

One simulation step remains one complete agent-governance-oversight decision cycle. It is not a fixed physical second/minute value. In the paper, map it illustratively to real time only if you explicitly state the assumed decision-cycle duration.
