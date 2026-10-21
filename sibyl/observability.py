"""Bounded, content-free operational metrics."""

from __future__ import annotations

import time
from collections import deque

BUCKETS = (0.1, 0.5, 1, 5, 30, 60, 600)
OUTCOMES = ("completed", "failed", "cancelled", "timeout", "rejected")


class RuntimeMetrics:
    def __init__(self):
        self.started = time.monotonic()
        self.tools = {}
        self.events = deque(maxlen=64)

    def record(self, tool: str, outcome: str, duration: float) -> None:
        if tool not in self.tools and len(self.tools) >= 32:
            tool = "other"
        if not tool.isascii() or not tool.replace("_", "").isalnum() or len(tool) > 48:
            tool = "other"
        value = self.tools.setdefault(
            tool,
            {
                "outcomes": dict.fromkeys(OUTCOMES, 0),
                "seconds": 0.0,
                "buckets": [0] * len(BUCKETS),
            },
        )
        value["outcomes"][outcome] += 1
        value["seconds"] += duration
        for index, bound in enumerate(BUCKETS):
            value["buckets"][index] += int(duration <= bound)
        self.events.append(
            {
                "tool": tool,
                "outcome": outcome,
                "duration_seconds": round(duration, 6),
                "timestamp": time.time(),
            }
        )

    def snapshot(self) -> dict:
        return {
            "uptime_seconds": round(time.monotonic() - self.started, 3),
            "tools": {
                name: {
                    "outcomes": dict(value["outcomes"]),
                    "duration_seconds_sum": round(value["seconds"], 6),
                }
                for name, value in self.tools.items()
            },
            "recent_events": list(self.events),
        }

    def prometheus(self, snapshot: dict) -> str:
        lines = [
            "# TYPE sibyl_runtime_up gauge",
            f"sibyl_runtime_up {int(not snapshot['closed'])}",
            "# TYPE sibyl_active_calls gauge",
            f"sibyl_active_calls {snapshot['active_calls']}",
            "# TYPE sibyl_capacity gauge",
            f"sibyl_capacity {snapshot['capacity']}",
            "# TYPE sibyl_tool_calls_total counter",
            "# TYPE sibyl_tool_duration_seconds histogram",
        ]
        for name, value in sorted(self.tools.items()):
            for outcome, count in value["outcomes"].items():
                lines.append(
                    f'sibyl_tool_calls_total{{tool="{name}",outcome="{outcome}"}} {count}'
                )
            for bound, count in zip(BUCKETS, value["buckets"]):
                lines.append(
                    f'sibyl_tool_duration_seconds_bucket{{tool="{name}",le="{bound}"}} {count}'
                )
            count = sum(value["outcomes"].values())
            lines.extend(
                [
                    f'sibyl_tool_duration_seconds_bucket{{tool="{name}",le="+Inf"}} {count}',
                    f'sibyl_tool_duration_seconds_count{{tool="{name}"}} {count}',
                    f'sibyl_tool_duration_seconds_sum{{tool="{name}"}} {value["seconds"]:.6f}',
                ]
            )
        lines.append("# TYPE sibyl_research_jobs gauge")
        for state, count in sorted(snapshot["jobs"].get("states", {}).items()):
            lines.append(f'sibyl_research_jobs{{state="{state}"}} {count}')
        lines.extend(
            [
                "# TYPE sibyl_store_healthy gauge",
                f"sibyl_store_healthy {int(snapshot['jobs']['healthy'])}",
            ]
        )
        for name, value in {
            "cache_entries": snapshot["cache"]["entries"],
            "cache_inflight": snapshot["cache"]["inflight"],
            "store_initialized": int(snapshot["jobs"]["initialized"]),
            "store_bytes": snapshot["jobs"].get("database_bytes", 0),
            "partial_reports": snapshot["jobs"].get("partial_reports", 0),
        }.items():
            lines.extend([f"# TYPE sibyl_{name} gauge", f"sibyl_{name} {value}"])
        return "\n".join(lines) + "\n"
