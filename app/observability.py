from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "data" / "metrics.jsonl"
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def record_metric(event: str, payload: Dict[str, Any]) -> None:
    entry = {"ts": time.time(), "event": event, **payload}
    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@contextmanager
def timed(event: str, extra: Optional[Dict[str, Any]] = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur_ms = (time.perf_counter() - start) * 1000.0
        record_metric(event, {"latency_ms": dur_ms, **(extra or {})})
