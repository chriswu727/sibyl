"""Bounded retrieval traces for MCP hosts that perform the reasoning."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from .evidence import (
    EvidenceLoop,
    EvidenceLoopAction,
    EvidenceLoopDiagnostics,
    EvidenceLoopStatus,
    EvidenceLoopStep,
    EvidenceLoopStepSummary,
    SourceBundle,
)
from .retrieval import query_requires_decomposition


Gather = Callable[[str, int, int, str, bool], Awaitable[SourceBundle]]


@dataclass
class _LoopState:
    loop_id: str
    question: str
    max_steps: int
    max_sources: int
    chars_per_source: int
    ranker: str
    render_thin_pages: bool
    created_at: float
    updated_at: float
    steps: list[EvidenceLoopStep] = field(default_factory=list)
    status: EvidenceLoopStatus = "active"
    next_action: EvidenceLoopAction = "decompose_query"
    supporting_step_ids: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EvidenceLoopManager:
    def __init__(self, *, ttl_seconds: int = 600, max_entries: int = 64):
        self.ttl_seconds = max(60, ttl_seconds)
        self.max_entries = max(1, max_entries)
        self._loops: OrderedDict[str, _LoopState] = OrderedDict()

    def clear(self) -> None:
        self._loops.clear()

    def _prune(self, now: float, *, reserve_entry: bool = False) -> None:
        expired = [
            loop_id
            for loop_id, state in self._loops.items()
            if now - state.updated_at >= self.ttl_seconds and not state.lock.locked()
        ]
        for loop_id in expired:
            self._loops.pop(loop_id, None)
        target_size = self.max_entries - 1 if reserve_entry else self.max_entries
        while len(self._loops) > target_size:
            removable = next(
                (
                    loop_id
                    for loop_id, state in self._loops.items()
                    if not state.lock.locked()
                ),
                None,
            )
            if removable is None:
                break
            self._loops.pop(removable, None)

    def _snapshot(
        self,
        state: _LoopState,
        *,
        current_step: Optional[EvidenceLoopStep] = None,
        error: str = "",
    ) -> EvidenceLoop:
        age = max(0, round(time.monotonic() - state.updated_at))
        return EvidenceLoop(
            schema_version="1.0",
            loop_id=state.loop_id,
            question=state.question,
            status=state.status,
            steps=[
                EvidenceLoopStepSummary(
                    step_id=step.step_id,
                    query=step.query,
                    bundle_id=step.bundle.bundle_id,
                    status=step.bundle.status,
                    evidence_sufficiency=step.bundle.diagnostics.evidence_sufficiency,
                    recommended_action=step.bundle.diagnostics.recommended_action,
                )
                for step in state.steps
            ],
            current_step=current_step,
            next_action=state.next_action,
            diagnostics=EvidenceLoopDiagnostics(
                max_steps=state.max_steps,
                retrieval_calls=len(state.steps),
                remaining_steps=max(0, state.max_steps - len(state.steps)),
                expires_in_seconds=max(0, self.ttl_seconds - age),
            ),
            supporting_step_ids=list(state.supporting_step_ids),
            error=error,
        )

    def invalid(self, question: str, error: str) -> EvidenceLoop:
        return EvidenceLoop(
            schema_version="1.0",
            loop_id="",
            question=question,
            status="invalid_request",
            steps=[],
            current_step=None,
            next_action="revise_request",
            diagnostics=EvidenceLoopDiagnostics(0, 0, 0, 0),
            error=error,
        )

    def failed(self, question: str, error: str) -> EvidenceLoop:
        return EvidenceLoop(
            schema_version="1.0",
            loop_id="",
            question=question,
            status="failed",
            steps=[],
            current_step=None,
            next_action="retry",
            diagnostics=EvidenceLoopDiagnostics(0, 0, 0, 0),
            error=error,
        )

    def _apply_bundle_action(self, state: _LoopState, bundle: SourceBundle) -> None:
        action = bundle.diagnostics.recommended_action
        remaining = state.max_steps - len(state.steps)
        if action == "synthesize":
            state.status = "active"
            state.next_action = "continue_or_finalize"
        elif action == "retry":
            state.status = "active" if remaining else "failed"
            state.next_action = "retry" if remaining else "none"
        elif action in {"refine_query", "decompose_query"}:
            state.status = "active" if remaining else "budget_exhausted"
            state.next_action = action if remaining else "none"
        elif action == "revise_request":
            state.status = "invalid_request"
            state.next_action = "revise_request"
        else:
            state.status = "failed"
            state.next_action = "none"

    async def start(
        self,
        question: str,
        *,
        max_steps: int,
        max_sources: int,
        chars_per_source: int,
        ranker: str,
        render_thin_pages: bool,
        gather: Gather,
    ) -> EvidenceLoop:
        clean_question = str(question or "").strip()
        if not clean_question:
            return self.invalid(clean_question, "question must not be empty.")
        if len(clean_question) > 1000:
            return self.invalid(clean_question, "question must be at most 1000 characters.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            return self.invalid(clean_question, "max_steps must be an integer from 1 to 4.")
        if max_steps < 1 or max_steps > 4:
            return self.invalid(clean_question, "max_steps must be between 1 and 4.")

        now = time.monotonic()
        self._prune(now, reserve_entry=True)
        if len(self._loops) >= self.max_entries:
            return self.failed(
                clean_question,
                "Evidence-loop capacity is busy. Retry after an active call completes.",
            )
        state = _LoopState(
            loop_id=f"el_{uuid4().hex}",
            question=clean_question,
            max_steps=max_steps,
            max_sources=max_sources,
            chars_per_source=chars_per_source,
            ranker=ranker,
            render_thin_pages=render_thin_pages,
            created_at=now,
            updated_at=now,
        )
        self._loops[state.loop_id] = state
        if query_requires_decomposition(clean_question):
            return self._snapshot(state)

        bundle = await gather(
            clean_question,
            max_sources,
            chars_per_source,
            ranker,
            render_thin_pages,
        )
        current_step = EvidenceLoopStep("E1", clean_question, bundle)
        state.steps.append(current_step)
        state.updated_at = time.monotonic()
        if bundle.diagnostics.recommended_action == "synthesize":
            state.status = "ready"
            state.next_action = "synthesize"
            state.supporting_step_ids = ["E1"]
        else:
            self._apply_bundle_action(state, bundle)
        return self._snapshot(state, current_step=current_step)

    async def advance(
        self,
        loop_id: str,
        *,
        query: str,
        finish: bool,
        supporting_step_ids: Optional[list[str]],
        gather: Gather,
    ) -> EvidenceLoop:
        now = time.monotonic()
        self._prune(now)
        state = self._loops.get(str(loop_id or "").strip())
        if state is None:
            return self.invalid(
                "",
                "Unknown or expired loop_id. Start a new evidence loop.",
            )
        self._loops.move_to_end(state.loop_id)

        async with state.lock:
            state.updated_at = time.monotonic()
            if finish:
                return self._finish(state, query, supporting_step_ids or [])
            return await self._retrieve_step(state, query, gather)

    def _finish(
        self,
        state: _LoopState,
        query: str,
        supporting_step_ids: list[str],
    ) -> EvidenceLoop:
        if str(query or "").strip():
            return self._snapshot(
                state,
                error="query must be empty when finish is true.",
            )
        if state.status == "ready":
            return self._snapshot(state)
        unique_ids = list(dict.fromkeys(str(value).strip() for value in supporting_step_ids))
        if not unique_ids or any(not value for value in unique_ids):
            return self._snapshot(
                state,
                error="supporting_step_ids must name at least one synthesis-ready step.",
            )
        steps = {step.step_id: step for step in state.steps}
        unknown = [step_id for step_id in unique_ids if step_id not in steps]
        if unknown:
            return self._snapshot(
                state,
                error=f"Unknown supporting step: {unknown[0]}.",
            )
        unsafe = [
            step_id
            for step_id in unique_ids
            if steps[step_id].bundle.diagnostics.recommended_action != "synthesize"
        ]
        if unsafe:
            return self._snapshot(
                state,
                error=f"Step {unsafe[0]} is not synthesis-ready.",
            )
        state.status = "ready"
        state.next_action = "synthesize"
        state.supporting_step_ids = unique_ids
        return self._snapshot(state)

    async def _retrieve_step(
        self,
        state: _LoopState,
        query: str,
        gather: Gather,
    ) -> EvidenceLoop:
        if state.status == "ready":
            return self._snapshot(state, error="This evidence loop is already ready.")
        if len(state.steps) >= state.max_steps:
            state.status = "budget_exhausted"
            state.next_action = "none"
            return self._snapshot(state, error="The retrieval-step budget is exhausted.")
        clean_query = str(query or "").strip()
        if not clean_query:
            return self._snapshot(state, error="query must not be empty.")
        if len(clean_query) > 1000:
            return self._snapshot(state, error="query must be at most 1000 characters.")
        if query_requires_decomposition(clean_query):
            state.next_action = "decompose_query"
            return self._snapshot(
                state,
                error="Follow-up queries must be atomic; decompose this query first.",
            )
        previous_queries = {step.query.casefold() for step in state.steps}
        if clean_query.casefold() in previous_queries:
            return self._snapshot(state, error="Follow-up queries must not repeat a prior step.")

        bundle = await gather(
            clean_query,
            state.max_sources,
            state.chars_per_source,
            state.ranker,
            state.render_thin_pages,
        )
        step_id = f"E{len(state.steps) + 1}"
        current_step = EvidenceLoopStep(step_id, clean_query, bundle)
        state.steps.append(current_step)
        state.updated_at = time.monotonic()
        self._apply_bundle_action(state, bundle)
        return self._snapshot(state, current_step=current_step)
