"""Per-server application state and bounded tool execution."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from .async_cache import AsyncSingleFlightTTL, CapacityError
from .config import Config
from .evidence import SourceBundle
from .observability import RuntimeMetrics
from .storage import default_output_root

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimePolicy:
    max_active: int = 8
    tool_timeout: float = 60.0
    report_timeout: float = 600.0
    max_query_chars: int = 1000
    output_root: Path = field(default_factory=default_output_root)
    monitor_port: int | None = None

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.max_active,
                self.tool_timeout,
                self.report_timeout,
                self.max_query_chars,
            )
        ) or (
            min(
                self.max_active,
                self.tool_timeout,
                self.report_timeout,
                self.max_query_chars,
            )
            <= 0
        ):
            raise ValueError("Runtime limits must be positive")


class SibylRuntime:
    """One trusted local MCP host owns one runtime; not a multi-tenant boundary."""

    def __init__(
        self, policy: RuntimePolicy | None = None, config: Config | None = None
    ):
        self.policy = policy or RuntimePolicy()
        self._config = config
        self.bundle_cache = AsyncSingleFlightTTL[tuple, SourceBundle](
            30,
            64,
            max_inflight=self.policy.max_active,
            operation_timeout=self.policy.tool_timeout,
        )
        self._report_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.closed = False
        self.completed = 0
        self.failed = 0
        self.cancelled = 0
        self._research_jobs = None
        self.metrics = RuntimeMetrics()
        self.monitor = None

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self.closed:
                raise RuntimeError("Server runtime is closed.")
            if self.policy.monitor_port is not None and self.monitor is None:
                from .monitoring import MonitorServer

                monitor = MonitorServer(
                    self,
                    self.policy.monitor_port,
                    os.environ.get("SIBYL_MONITOR_TOKEN", ""),
                )
                await monitor.start()
                self.monitor = monitor
                logger.info(
                    json.dumps(
                        {
                            "event": "monitor_started",
                            "host": "127.0.0.1",
                            "port": monitor.port,
                        }
                    )
                )

    def snapshot(self) -> dict:
        jobs = {
            "healthy": True,
            "initialized": self._research_jobs is not None,
            "states": {},
        }
        if self._research_jobs is not None:
            try:
                jobs.update(self._research_jobs.store.summary())
            except Exception:
                jobs["healthy"] = False
        return {
            "schema_version": "1.0",
            "closed": self.closed,
            "active_calls": len(self._tasks),
            "capacity": self.policy.max_active,
            "cache": self.bundle_cache.snapshot(),
            "jobs": jobs,
            **self.metrics.snapshot(),
        }

    @property
    def research_jobs(self):
        if self.closed:
            raise RuntimeError("Server runtime is closed.")
        if self._research_jobs is None:
            from .research_jobs import ResearchJobs

            self._research_jobs = ResearchJobs(self)
        return self._research_jobs

    @property
    def config(self) -> Config:
        if self._config is None:
            path = os.environ.get("SIBYL_CONFIG")
            if path:
                self._config = Config.from_yaml(path)
            else:
                self._config = Config.from_env()
        return self._config

    def validate_query(self, query: str) -> None:
        if len(query) > self.policy.max_query_chars:
            raise ValueError(
                f"Query must be at most {self.policy.max_query_chars} characters."
            )

    async def execute(self, name: str, operation: Callable[[], Awaitable[T]]) -> T:
        if self.closed:
            self.metrics.record(name, "rejected", 0)
            raise RuntimeError("Server runtime is closed.")
        if len(self._tasks) >= self.policy.max_active:
            self.metrics.record(name, "rejected", 0)
            raise CapacityError("Server capacity is busy. Retry later.")
        stateful = name == "save_report"
        if stateful and self._report_lock.locked():
            self.metrics.record(name, "rejected", 0)
            raise CapacityError(
                "Another report operation is active. Retry after it completes."
            )
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        locked = False
        started, outcome = time.monotonic(), "failed"
        try:
            if stateful:
                await self._report_lock.acquire()
                locked = True
            timeout = self.policy.tool_timeout
            result = await asyncio.wait_for(operation(), timeout)
            self.completed += 1
            outcome = "completed"
            return result
        except asyncio.CancelledError:
            self.cancelled += 1
            outcome = "cancelled"
            raise
        except asyncio.TimeoutError:
            self.failed += 1
            outcome = "timeout"
            raise
        except Exception:
            self.failed += 1
            raise
        finally:
            if locked:
                self._report_lock.release()
            self._tasks.discard(task)
            elapsed = time.monotonic() - started
            self.metrics.record(name, outcome, elapsed)
            logger.info(
                json.dumps(
                    {
                        "event": "tool_finished",
                        "tool": name,
                        "outcome": outcome,
                        "duration_seconds": round(elapsed, 6),
                        "active": len(self._tasks),
                    }
                )
            )

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            self.closed = True
            if self.monitor is not None:
                await self.monitor.close()
                self.monitor = None
        tasks = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._research_jobs is not None:
            await self._research_jobs.close()
        await self.bundle_cache.aclose()


default_runtime = SibylRuntime()
current_runtime: ContextVar[SibylRuntime] = ContextVar(
    "sibyl_runtime", default=default_runtime
)


def get_runtime() -> SibylRuntime:
    return current_runtime.get()
