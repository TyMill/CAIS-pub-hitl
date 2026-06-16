[![DOI](https://zenodo.org/badge/1186199729.svg)](https://doi.org/10.5281/zenodo.19110440)

# MARIS-AI (v0.4.0) — CAIS + Risk-Aware Human-in-the-Loop Governance

End-to-end, self-contained experimental stack for:
- multi-agent maritime-inspired encounters (2D),
- governance-constrained decision execution (**G**),
- risk-aware human-in-the-loop governance (**HITL-G**),
- audit trace semantics (**Φ**) with hash chaining,
- replayability verification (**Ψ**),
- federated learning (FedAvg),
- results-ready CSV/JSON exports.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Standard CAIS sweep

```bash
maris-cais-sweep --episodes 30 --steps 200 --agents 5 --seed 42 --scenario crossing --centralized --projection slsqp
```

## Risk-aware HITL sweep

```bash
maris-cais-hitl \
  --scenario bottleneck \
  --episodes 20 \
  --steps 100 \
  --agents 5 \
  --seed 42 \
  --modes adaptive_hitl \
  --noise-std 0.10 \
  --risk-threshold 0.40 \
  --risk-threshold-low 0.20 \
  --cooldown-steps 5 \
  --reliability 0.95 \
  --delay-steps 0 \
  --outputs outputs_hitl_v04_test
```

## Paper-ready threshold sweep

```bash
for TH in 0.10 0.20 0.30 0.40 0.50 0.60; do
  maris-cais-hitl \
    --scenario bottleneck \
    --episodes 50 \
    --steps 150 \
    --agents 5 \
    --seed 42 \
    --modes adaptive_hitl \
    --noise-std 0.10 \
    --risk-threshold $TH \
    --risk-threshold-low $(python - <<PY
print(float("$TH")/2)
PY
) \
    --cooldown-steps 5 \
    --reliability 0.95 \
    --delay-steps 0 \
    --outputs outputs_hitl_threshold_$TH
done
```

## Operator delay sweep

```bash
for DELAY in 0 1 3 5 10; do
  maris-cais-hitl \
    --scenario bottleneck \
    --episodes 50 \
    --steps 150 \
    --agents 5 \
    --seed 42 \
    --modes adaptive_hitl \
    --noise-std 0.10 \
    --risk-threshold 0.40 \
    --risk-threshold-low 0.20 \
    --cooldown-steps 5 \
    --reliability 0.95 \
    --delay-steps $DELAY \
    --outputs outputs_hitl_delay_$DELAY
done
```

## Operator reliability sweep

```bash
for REL in 1.00 0.95 0.90 0.75 0.50; do
  maris-cais-hitl \
    --scenario bottleneck \
    --episodes 50 \
    --steps 150 \
    --agents 5 \
    --seed 42 \
    --modes adaptive_hitl \
    --noise-std 0.10 \
    --risk-threshold 0.40 \
    --risk-threshold-low 0.20 \
    --cooldown-steps 5 \
    --reliability $REL \
    --delay-steps 0 \
    --outputs outputs_hitl_reliability_$REL
done
```

## Integrity replay (Ψ)

```bash
maris-cais-replay outputs/<run_id>/
```

## Main HITL modes

- `cais_only` — CAIS action without human intervention.
- `human_approval` — human confirms or misses CAIS action at every step.
- `human_override` — human may override unsafe behaviour.
- `adaptive_hitl` — risk-aware activation with composite risk, hysteresis and cooldown.

## New in v0.4.0

The previous HITL version used a raw max-residual trigger, which behaved almost binarily. v0.4.0 adds:

- `human/risk.py` — composite risk score: separation, congestion, speed and uncertainty.
- `human/trigger.py` — stateful hysteresis and cooldown trigger.
- extended HITL metrics: `risk_mean`, `risk_p95`, component risks, `cooldown_skips`, `trigger_active_rate`, `false_intervention_rate`.

Each run exports `metrics.json`, `trace.jsonl`, `trace_ok.json`, and a sweep-level `hitl_results.csv`.
