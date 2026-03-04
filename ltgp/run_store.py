from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def _json_default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def stable_hash(d: Dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


def make_run_dir(base: str, eq_id: str, config: Dict[str, Any]) -> str:
    os.makedirs(base, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    h = stable_hash({"eq_id": eq_id, **config})
    name = f"{ts}_{eq_id}_{h}"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, data: Dict[str, Any], filename: str = "config.json") -> str:
    full = os.path.join(path, filename)
    with open(full, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
    return full


def write_text(path: str, text: str, filename: str = "summary.txt") -> str:
    full = os.path.join(path, filename)
    with open(full, "w", encoding="utf-8") as f:
        f.write(text)
    return full


def save_npy(path: str, arr, filename: str) -> str:
    import numpy as np
    full = os.path.join(path, filename)
    np.save(full, arr)
    return full


def save_fig(path: str, fig, filename: str) -> str:
    full = os.path.join(path, filename)
    fig.savefig(full, dpi=200, bbox_inches="tight")
    return full