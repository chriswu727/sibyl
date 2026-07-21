"""Model-free evaluation for fixed host-planned evidence loops."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Awaitable, Callable, Sequence

from .api import gather_bundle
from .evidence import SourceBundle
from .evidence_loop import EvidenceLoopManager


@dataclass(frozen=True)
class EvidenceLoopEvalCase:
    case_id: str
    question: str
    gold: str
    aliases: list[str]
    queries: list[str]


@dataclass(frozen=True)
class EvidenceLoopEvalStep:
    step_id: str
    query: str
    status: str
    recommended_action: str
    source_count: int
    latency_ms: int


@dataclass(frozen=True)
class EvidenceLoopEvalCaseResult:
    case_id: str
    initial_query_decomposed: bool
    planned_queries_completed: bool
    loop_status: str
    answer_in_evidence: bool
    passed: bool
    error: str
    steps: list[EvidenceLoopEvalStep]


@dataclass(frozen=True)
class EvidenceLoopEvalResult:
    total_cases: int
    decomposed_cases: int
    completed_plan_cases: int
    ready_cases: int
    answer_hits: int
    passed_cases: int
    decomposition_rate: float
    plan_execution_rate: float
    ready_rate: float
    answer_coverage: float
    pass_rate: float
    cases: list[EvidenceLoopEvalCaseResult]

    def to_dict(self):
        return asdict(self)


Gather = Callable[..., Awaitable[SourceBundle]]


def _normalized(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_equivalent = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[^\W_]+", ascii_equivalent.casefold()))


def _contains_candidate(evidence: str, candidate: str) -> bool:
    normalized_candidate = _normalized(candidate)
    normalized_evidence = _normalized(evidence)
    if not normalized_candidate:
        return False
    if normalized_candidate in normalized_evidence:
        return True
    candidate_terms = normalized_candidate.split()
    return len(candidate_terms) <= 5 and set(candidate_terms) <= set(
        normalized_evidence.split()
    )


async def evaluate_evidence_loops(
    cases: Sequence[EvidenceLoopEvalCase],
    *,
    max_sources: int = 10,
    concurrency: int = 1,
    gather: Gather = gather_bundle,
) -> EvidenceLoopEvalResult:
    if not cases:
        raise ValueError("At least one evidence-loop case is required.")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def evaluate(case: EvidenceLoopEvalCase) -> EvidenceLoopEvalCaseResult:
        manager = EvidenceLoopManager()

        async def retrieve(query, sources, chars, ranker, render_thin_pages):
            return await gather(
                query,
                max_sources=sources,
                chars_per_source=chars,
                ranker=ranker,
                render_thin_pages=render_thin_pages,
            )

        bundles: list[SourceBundle] = []
        steps: list[EvidenceLoopEvalStep] = []
        async with semaphore:
            loop = await manager.start(
                case.question,
                max_steps=4,
                max_sources=max_sources,
                chars_per_source=7000,
                ranker="lexical",
                render_thin_pages=False,
                gather=retrieve,
            )
            initial_query_decomposed = loop.current_step is None
            if loop.current_step is not None:
                bundles.append(loop.current_step.bundle)
            for query in case.queries:
                if loop.status == "ready":
                    break
                loop = await manager.advance(
                    loop.loop_id,
                    query=query,
                    finish=False,
                    supporting_step_ids=None,
                    gather=retrieve,
                )
                if loop.current_step is None:
                    continue
                bundle = loop.current_step.bundle
                bundles.append(bundle)
                steps.append(
                    EvidenceLoopEvalStep(
                        step_id=loop.current_step.step_id,
                        query=query,
                        status=bundle.status,
                        recommended_action=bundle.diagnostics.recommended_action,
                        source_count=len(bundle.sources),
                        latency_ms=bundle.diagnostics.latency_ms,
                    )
                )
            supporting_step_ids = [step.step_id for step in loop.steps]
            loop = await manager.advance(
                loop.loop_id,
                query="",
                finish=True,
                supporting_step_ids=supporting_step_ids,
                gather=retrieve,
            )

        evidence = "\n".join(
            passage.text
            for bundle in bundles
            for source in bundle.sources
            for passage in source.evidence
        )
        candidates = [case.gold, *case.aliases]
        answer_in_evidence = any(
            _contains_candidate(evidence, candidate) for candidate in candidates
        )
        planned_queries_completed = len(steps) == len(case.queries)
        passed = (
            initial_query_decomposed
            and planned_queries_completed
            and loop.status == "ready"
            and answer_in_evidence
        )
        return EvidenceLoopEvalCaseResult(
            case_id=case.case_id,
            initial_query_decomposed=initial_query_decomposed,
            planned_queries_completed=planned_queries_completed,
            loop_status=loop.status,
            answer_in_evidence=answer_in_evidence,
            passed=passed,
            error=loop.error,
            steps=steps,
        )

    results = await asyncio.gather(*(evaluate(case) for case in cases))
    total = len(results)
    decomposed = sum(result.initial_query_decomposed for result in results)
    completed_plans = sum(result.planned_queries_completed for result in results)
    ready = sum(result.loop_status == "ready" for result in results)
    answer_hits = sum(result.answer_in_evidence for result in results)
    passed = sum(result.passed for result in results)
    return EvidenceLoopEvalResult(
        total_cases=total,
        decomposed_cases=decomposed,
        completed_plan_cases=completed_plans,
        ready_cases=ready,
        answer_hits=answer_hits,
        passed_cases=passed,
        decomposition_rate=round(decomposed / total, 6),
        plan_execution_rate=round(completed_plans / total, 6),
        ready_rate=round(ready / total, 6),
        answer_coverage=round(answer_hits / total, 6),
        pass_rate=round(passed / total, 6),
        cases=results,
    )
