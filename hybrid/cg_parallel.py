"""Order-preserving parallel map for the Continuous Gain steps that are many INDEPENDENT native renders (one probe per capture, one render
per capture per training segment, one comparison per position). The native renderer runs as a separate single-threaded process per call,
so threads are enough; results come back in input order, so every output is identical to the serial computation."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    """NAM_MIXER_CG_WORKERS overrides; otherwise half the cores, capped at 6 (renders are CPU-bound and memory-hungry)."""
    env = os.environ.get("NAM_MIXER_CG_WORKERS", "").strip()
    if env.isdigit() and int(env) >= 1:
        return int(env)
    return max(1, min(6, (os.cpu_count() or 2) // 2))


def pmap(fn: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> list[R]:
    items = list(items)
    n = min(workers or default_workers(), len(items))
    if n <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(fn, items))            # pool.map preserves order and re-raises the first exception
