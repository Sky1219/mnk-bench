"""Saving and loading results as plain JSON/CSV so plots can be redrawn
without re-running experiments."""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any


def _clean(obj: Any) -> Any:
    """Make ``obj`` strict-JSON safe (NaN/inf -> None, tuples -> lists)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _clean(obj.to_dict())
    if hasattr(obj, "value") and obj.__class__.__module__ != "builtins":  # Enums
        return obj.value
    return obj


def metadata(**extra: Any) -> dict:
    from . import __version__

    return {
        "mnkbench_version": __version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(data), indent=1, allow_nan=False))
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _clean(v) for k, v in r.items()})
    return path
