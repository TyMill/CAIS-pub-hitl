from __future__ import annotations
import json, hashlib
from typing import Any

def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)

def stable_hash(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
