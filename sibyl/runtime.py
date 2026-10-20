"""Per-server application state and bounded tool execution."""

from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from .async_cache import AsyncSingleFlightTTL, CapacityError
from .config import Config
from .evidence import SourceBundle
from .evidence_loop import EvidenceLoopManager
from .report_models import ResearchReport

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimePolicy:
    max_active: int = 8
    tool_timeout: float = 60.0
    report_timeout: float = 600.0
    max_query_chars: int = 1000
    output_root: Path = Path("reports")

    def __post_init__(self) -> None:
        if (
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
        self.policy = policy or RuntimePolicy(
            output_root=Path(os.environ.get("SIBYL_OUTPUT_ROOT", "reports")).absolute()
        )
        self._config = config
        self.bundle_cache = AsyncSingleFlightTTL[tuple, SourceBundle](
            30,
            64,
            max_inflight=self.policy.max_active,
            operation_timeout=self.policy.tool_timeout,
        )
        self.evidence_loops = EvidenceLoopManager()
        self.last_report: ResearchReport | None = None
        self._report_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.closed = False
        self.completed = 0
        self.failed = 0
        self.cancelled = 0
        self._research_jobs = None

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
                self._config = Config.from_env(
                    model=os.environ.get("SIBYL_MODEL", ""),
                    api_key=os.environ.get("SIBYL_API_KEY", ""),
                    api_base=os.environ.get("SIBYL_API_BASE", ""),
                )
        return self._config

    def validate_query(self, query: str) -> None:
        if len(query) > self.policy.max_query_chars:
            raise ValueError(
                f"Query must be at most {self.policy.max_query_chars} characters."
            )

    async def execute(self, name: str, operation: Callable[[], Awaitable[T]]) -> T:
        if self.closed:
            raise RuntimeError("Server runtime is closed.")
        if len(self._tasks) >= self.policy.max_active:
            raise CapacityError("Server capacity is busy. Retry later.")
        stateful = name in {"research", "save_report", "chart", "fetch_market_data"}
        if stateful and self._report_lock.locked():
            raise CapacityError(
                "Another report operation is active. Retry after it completes."
            )
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        locked = False
        try:
            if stateful:
                await self._report_lock.acquire()
                locked = True
            timeout = (
                self.policy.report_timeout
                if name in {"research", "analyze", "compare", "swot", "timeline"}
                else self.policy.tool_timeout
            )
            result = await asyncio.wait_for(operation(), timeout)
            self.completed += 1
            return result
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        except Exception:
            self.failed += 1
            raise
        finally:
            if locked:
                self._report_lock.release()
            self._tasks.discard(task)
            logger.info(
                "tool_finished tool=%s completed=%d failed=%d cancelled=%d active=%d",
                name,
                self.completed,
                self.failed,
                self.cancelled,
                len(self._tasks),
            )

    async def aclose(self) -> None:
        self.closed = True
        tasks = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._research_jobs is not None:
            await self._research_jobs.close()
        await self.bundle_cache.aclose()
        self.evidence_loops.clear()
        self.last_report = None


default_runtime = SibylRuntime()
current_runtime: ContextVar[SibylRuntime] = ContextVar(
    "sibyl_runtime", default=default_runtime
)


def get_runtime() -> SibylRuntime:
    return current_runtime.get()
