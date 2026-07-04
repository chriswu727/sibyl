#!/usr/bin/env python
"""Benchmark the research pipeline across depths + fast mode.

Usage:
    export DEEPSEEK_API_KEY=sk-...      # or any supported provider key
    python scripts/bench.py [--query "..."] [--trials 2]

Prints per-run wall time and per-phase timing so you can see where the time
goes. See docs/PERFORMANCE.md for the optimization history and current floor.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from sibyl.config import Config
from sibyl.researcher import Researcher

DEFAULT_QUERY = "Will NVIDIA maintain its AI chip dominance through 2027?"


async def _one(query: str, depth: int, fast: bool):
    cfg = Config.from_env()
    cfg.fast = fast
    r = Researcher(cfg)
    steps: list[tuple[float, str]] = []
    t0 = time.time()
    rep = await r.research(query, depth=depth, on_progress=lambda m: steps.append((time.time() - t0, m)))
    return time.time() - t0, rep, steps


def _phase_table(steps):
    """Turn progress timestamps into per-phase durations."""
    out = []
    for i, (t, msg) in enumerate(steps):
        end = steps[i + 1][0] if i + 1 < len(steps) else t
        out.append((round(end - t, 1), msg[:48]))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--phases", action="store_true", help="print per-phase timing for depth-2")
    args = ap.parse_args()

    configs = [("depth-1", 1, False), ("depth-2", 2, False),
               ("depth-2 fast", 2, True), ("depth-3", 3, False)]

    for label, depth, fast in configs:
        times = []
        last_steps = None
        for _ in range(args.trials):
            dt, rep, steps = await _one(args.query, depth, fast)
            times.append(dt)
            last_steps = steps
            ok = len(rep.summary) > 200 and len(rep.key_findings) >= 3
            status = "OK" if ok else "FAIL"
        avg = sum(times) / len(times)
        spread = f" (min {min(times):.1f} max {max(times):.1f})" if len(times) > 1 else ""
        print(f"{label:14s} avg={avg:5.1f}s{spread}  [{status}]")
        if args.phases and label == "depth-2" and last_steps:
            for dur, msg in _phase_table(last_steps):
                print(f"    {dur:5.1f}s  {msg}")


if __name__ == "__main__":
    asyncio.run(main())
