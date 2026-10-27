"""Event-loop-local async single-flight cache."""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Generic, Hashable, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class _CacheState(Generic[K, V]):
    entries: OrderedDict[K, Tuple[float, V]] = field(default_factory=OrderedDict)
    inflight: Dict[K, asyncio.Task[V]] = field(default_factory=dict)
    waiters: Dict[K, int] = field(default_factory=dict)


class CapacityError(RuntimeError):
    """The bounded executor cannot admit another operation."""


class AsyncSingleFlightTTL(Generic[K, V]):
    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
        *,
        max_inflight: int = 8,
        operation_timeout: float = 60.0,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_inflight <= 0 or operation_timeout <= 0:
            raise ValueError("execution limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self.max_inflight = max_inflight
        self.operation_timeout = operation_timeout
        self._states = weakref.WeakKeyDictionary()
        self._states_lock = threading.Lock()

    def _state(self) -> _CacheState[K, V]:
        loop = asyncio.get_running_loop()
        with self._states_lock:
            state = self._states.get(loop)
            if state is None:
                state = _CacheState()
                self._states[loop] = state
            return state

    async def get_or_create(
        self,
        key: K,
        factory: Callable[[], Awaitable[V]],
        should_cache: Callable[[V], bool] = lambda value: True,
    ) -> V:
        state = self._state()
        cached = state.entries.get(key)
        if cached is not None:
            expires_at, value = cached
            if expires_at > self.clock():
                state.entries.move_to_end(key)
                return value
            del state.entries[key]

        task = state.inflight.get(key)
        if task is None:
            if len(state.inflight) >= self.max_inflight:
                raise CapacityError("Retrieval capacity is busy. Retry later.")
            task = asyncio.create_task(
                self._populate(state, key, factory, should_cache)
            )
            task.add_done_callback(self._observe_completion)
            state.inflight[key] = task
        state.waiters[key] = state.waiters.get(key, 0) + 1
        try:
            return await asyncio.shield(task)
        finally:
            remaining = state.waiters.get(key, 1) - 1
            if remaining:
                state.waiters[key] = remaining
            else:
                state.waiters.pop(key, None)
                if not task.done():
                    task.cancel()
                    # Keep the admission slot until the factory releases its resources.
                    await asyncio.gather(task, return_exceptions=True)

    async def aclose(self) -> None:
        state = self._state()
        tasks = list(state.inflight.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        state.entries.clear()

    def snapshot(self) -> dict:
        state = self._state()
        return {
            "entries": sum(
                expires > self.clock() for expires, _ in state.entries.values()
            ),
            "inflight": len(state.inflight),
            "waiters": sum(state.waiters.values()),
        }

    @staticmethod
    def _observe_completion(task: asyncio.Task[V]) -> None:
        if not task.cancelled():
            task.exception()

    async def _populate(
        self,
        state: _CacheState[K, V],
        key: K,
        factory: Callable[[], Awaitable[V]],
        should_cache: Callable[[V], bool],
    ) -> V:
        task = asyncio.current_task()
        try:
            value = await asyncio.wait_for(factory(), self.operation_timeout)
            if should_cache(value):
                state.entries[key] = (self.clock() + self.ttl_seconds, value)
                state.entries.move_to_end(key)
                while len(state.entries) > self.max_entries:
                    state.entries.popitem(last=False)
            return value
        finally:
            if state.inflight.get(key) is task:
                del state.inflight[key]
