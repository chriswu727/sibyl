"""Live, model-free evaluation for keyless retrieval."""
from __future__ import annotations

import asyncio
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, List, Sequence

from .api import gather_bundle
from .evidence import SourceBundle


@dataclass(frozen=True)
class LiveRetrievalCase:
    case_id: str
    question: str
    gold: str
    aliases: List[str]
    case_type: str = ""

    @property
    def expects_answer(self) -> bool:
        return self.gold.strip().upper() != "NO_ANSWER"


@dataclass(frozen=True)
class LiveRetrievalRun:
    status: str
    evidence_sufficiency: str
    answer_in_evidence: bool
    refutation_in_evidence: bool
    safe_trap_outcome: bool
    source_count: int
    latency_ms: int
    error: str = ""
    search_results: int = 0
    urls_attempted: int = 0
    pages_scraped: int = 0
    scrape_failures: int = 0
    wikipedia_fallbacks: int = 0
    metadata_fallbacks: int = 0
    search_providers: List[str] = field(default_factory=list)
    max_source_query_term_coverage: float | None = None
    sufficiency_reasons: List[str] = field(default_factory=list)
    query_complexity: str = "not_assessed"
    recommended_action: str = "not_assessed"
    refinement_searches: int = 0
    refinement_failures: int = 0


@dataclass(frozen=True)
class LiveRetrievalCaseResult:
    case_id: str
    case_type: str
    expects_answer: bool
    stable: bool
    runs: List[LiveRetrievalRun]


@dataclass(frozen=True)
class LiveRetrievalEvalResult:
    total_cases: int
    answerable_cases: int
    trap_cases: int
    answer_coverage: float
    trap_safe_rate: float
    stable_case_rate: float
    status_ok_rate: float
    answerable_ready_rate: float
    ready_answer_precision: float
    p50_latency_ms: int
    p95_latency_ms: int
    cases: List[LiveRetrievalCaseResult]

    def to_dict(self):
        return asdict(self)


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


def _bundle_evidence(bundle: SourceBundle) -> str:
    return "\n".join(
        passage.text
        for source in bundle.sources
        for passage in source.evidence
    )


def evaluate_bundle(
    case: LiveRetrievalCase,
    bundle: SourceBundle,
) -> LiveRetrievalRun:
    evidence = _bundle_evidence(bundle)
    answer_candidates = [case.gold, *case.aliases] if case.expects_answer else []
    answer_in_evidence = any(
        _contains_candidate(evidence, candidate) for candidate in answer_candidates
    )
    refutation_in_evidence = any(
        _contains_candidate(evidence, alias) for alias in case.aliases
    ) if not case.expects_answer else False
    safe_trap_outcome = (
        not case.expects_answer
        and (bundle.status != "ok" or refutation_in_evidence)
    )
    return LiveRetrievalRun(
        status=bundle.status,
        evidence_sufficiency=bundle.diagnostics.evidence_sufficiency,
        answer_in_evidence=answer_in_evidence,
        refutation_in_evidence=refutation_in_evidence,
        safe_trap_outcome=safe_trap_outcome,
        source_count=len(bundle.sources),
        latency_ms=bundle.diagnostics.latency_ms,
        search_results=bundle.diagnostics.search_results,
        urls_attempted=bundle.diagnostics.urls_attempted,
        pages_scraped=bundle.diagnostics.pages_scraped,
        scrape_failures=bundle.diagnostics.scrape_failures,
        wikipedia_fallbacks=bundle.diagnostics.wikipedia_fallbacks,
        metadata_fallbacks=bundle.diagnostics.metadata_fallbacks,
        search_providers=bundle.diagnostics.search_providers,
        max_source_query_term_coverage=(
            bundle.diagnostics.max_source_query_term_coverage
        ),
        sufficiency_reasons=bundle.diagnostics.sufficiency_reasons,
        query_complexity=bundle.diagnostics.query_complexity,
        recommended_action=bundle.diagnostics.recommended_action,
        refinement_searches=bundle.diagnostics.refinement_searches,
        refinement_failures=bundle.diagnostics.refinement_failures,
    )


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


async def evaluate_live_retrieval(
    cases: Sequence[LiveRetrievalCase],
    *,
    repeats: int = 1,
    concurrency: int = 2,
    max_sources: int = 10,
    gather: Callable[..., Awaitable[SourceBundle]] = gather_bundle,
    progress: Callable[[int, int, LiveRetrievalCaseResult], None] | None = None,
) -> LiveRetrievalEvalResult:
    if not cases:
        raise ValueError("At least one live retrieval case is required.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    completed = 0

    async def evaluate_case(case: LiveRetrievalCase) -> LiveRetrievalCaseResult:
        nonlocal completed
        runs = []
        for _ in range(repeats):
            async with semaphore:
                started = time.monotonic()
                try:
                    bundle = await gather(case.question, max_sources=max_sources)
                except Exception as exc:
                    runs.append(
                        LiveRetrievalRun(
                            status="failed",
                            evidence_sufficiency="not_assessed",
                            answer_in_evidence=False,
                            refutation_in_evidence=False,
                            safe_trap_outcome=not case.expects_answer,
                            source_count=0,
                            latency_ms=round((time.monotonic() - started) * 1000),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
            runs.append(evaluate_bundle(case, bundle))
        outcomes = {
            (run.status, run.answer_in_evidence, run.safe_trap_outcome)
            for run in runs
        }
        result = LiveRetrievalCaseResult(
            case_id=case.case_id,
            case_type=case.case_type,
            expects_answer=case.expects_answer,
            stable=len(outcomes) == 1,
            runs=runs,
        )
        completed += 1
        if progress is not None:
            progress(completed, len(cases), result)
        return result

    results = await asyncio.gather(*(evaluate_case(case) for case in cases))
    answerable = [result for result in results if result.expects_answer]
    traps = [result for result in results if not result.expects_answer]
    answer_hits = sum(
        all(run.answer_in_evidence for run in result.runs) for result in answerable
    )
    trap_safe = sum(
        all(run.safe_trap_outcome for run in result.runs) for result in traps
    )
    all_runs = [run for result in results for run in result.runs]
    status_ok_runs = sum(run.status == "ok" for run in all_runs)
    answerable_runs = [
        run for result in answerable for run in result.runs
    ]
    answerable_ok_runs = [
        run for run in answerable_runs if run.status == "ok"
    ]
    answerable_ready_runs = sum(
        run.answer_in_evidence for run in answerable_ok_runs
    )
    latencies = [run.latency_ms for run in all_runs]
    return LiveRetrievalEvalResult(
        total_cases=len(results),
        answerable_cases=len(answerable),
        trap_cases=len(traps),
        answer_coverage=round(answer_hits / len(answerable), 6) if answerable else 1.0,
        trap_safe_rate=round(trap_safe / len(traps), 6) if traps else 1.0,
        stable_case_rate=round(
            sum(result.stable for result in results) / len(results), 6
        ),
        status_ok_rate=round(status_ok_runs / len(all_runs), 6),
        answerable_ready_rate=round(
            answerable_ready_runs / len(answerable_runs), 6
        ) if answerable_runs else 1.0,
        ready_answer_precision=round(
            answerable_ready_runs / len(answerable_ok_runs), 6
        ) if answerable_ok_runs else 1.0,
        p50_latency_ms=_percentile(latencies, 0.5),
        p95_latency_ms=_percentile(latencies, 0.95),
        cases=results,
    )
