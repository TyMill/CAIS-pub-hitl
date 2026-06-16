from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from maris_ai.audit.hashing import stable_hash

@dataclass
class TraceRecord:
    t: int
    scenario: str
    state: Dict[str, Any]
    proposals: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    residuals: Dict[str, float]
    governance_meta: Dict[str, Any]
    seed: int
    policy_ids: List[str]
    C: Dict[str, Any]
    prev_hash: str
    this_hash: str

def make_trace_record(t: int, scenario: str, positions, velocities, proposals, actions,
                      residuals: Dict[str, float], governance_meta: Dict[str, Any],
                      seed: int, policy_ids: List[str], C: Dict[str, Any], prev_hash: str) -> TraceRecord:
    state_obj = {"positions": [p for p in positions], "velocities": [v for v in velocities], "t": t}
    payload = {
        "t": t, "scenario": scenario, "state": state_obj,
        "proposals": [{"v_cmd": p.v_cmd} for p in proposals],
        "actions": [{"v_exec": a} for a in actions],
        "residuals": residuals, "governance_meta": governance_meta,
        "seed": seed, "policy_ids": policy_ids, "C": C, "prev_hash": prev_hash,
    }
    this_hash = stable_hash(payload)
    return TraceRecord(t=t, scenario=scenario, state=state_obj,
                       proposals=payload["proposals"], actions=payload["actions"],
                       residuals={k: float(v) for k,v in residuals.items()},
                       governance_meta=governance_meta, seed=int(seed),
                       policy_ids=list(policy_ids), C=dict(C),
                       prev_hash=prev_hash, this_hash=this_hash)
