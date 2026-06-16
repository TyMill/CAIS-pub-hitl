from __future__ import annotations
from typing import List
from maris_ai.audit.trace import TraceRecord
from maris_ai.audit.hashing import stable_hash

def verify_hash_chain(records: List[TraceRecord]) -> bool:
    prev = "GENESIS"
    for r in records:
        if r.prev_hash != prev:
            return False
        payload = {
            "t": r.t, "scenario": r.scenario, "state": r.state,
            "proposals": r.proposals, "actions": r.actions,
            "residuals": r.residuals, "governance_meta": r.governance_meta,
            "seed": r.seed, "policy_ids": r.policy_ids, "C": r.C,
            "prev_hash": r.prev_hash,
        }
        if stable_hash(payload) != r.this_hash:
            return False
        prev = r.this_hash
    return True
