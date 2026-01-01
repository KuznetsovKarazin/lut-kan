# src/metrics/perf.py
from __future__ import annotations

import time
from typing import Callable, Dict


def measure_latency(fn: Callable[[], None], warmup_iters: int = 10, measure_iters: int = 100) -> Dict[str, float]:
    for _ in range(int(warmup_iters)):
        fn()

    t0 = time.perf_counter()
    for _ in range(int(measure_iters)):
        fn()
    t1 = time.perf_counter()

    total = t1 - t0
    per_iter = total / max(1, int(measure_iters))
    return {
        "total_s": float(total),
        "per_iter_ms": float(per_iter * 1000.0),
        "iters": float(measure_iters),
    }
