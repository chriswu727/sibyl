"""Deterministic next-search guidance for the host's own web search.

An MCP server cannot run the host's search tool, so Sibyl names the searches
that would close each question's gaps and explains why. Every suggestion comes
from the recorded research state: the host's own follow-ups, which questions
have no reading, whether only background sources support a question, whether
its evidence comes from one domain, missing counterevidence and recency. These
are templates and rules, not a model; the host decides which to run.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from .evidence import EvidenceSource
from .job_models import (
    QuestionPlan,
    RetrievalAttempt,
    SearchPurpose,
    SuggestedQuery,
    SuggestionKind,
)
from .queries import keyword_query, search_query_variants
from .source_assessment import BACKGROUND_KINDS
from .source_policy import source_kind

if TYPE_CHECKING:
    from .research_state import ResearchNotebook


def question_attempts(
    notebook: ResearchNotebook, question_id: str, question: str
) -> list[RetrievalAttempt]:
    return [
        a
        for a in notebook.attempts
        if a.get("question_id") == question_id
        or (not a.get("question_id") and a["query"] == question)
    ]


PER_QUESTION = 3
TOTAL = 10
_RECENCY_RE = re.compile(
    r"\b(?:latest|current|currently|now|today|recent|recently|this year)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _normalized(query: str) -> str:
    return " ".join(query.casefold().split())


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _core_query(question: str) -> str:
    """The question without question words and stop words, for a search box."""
    variants = search_query_variants(question)
    return variants[-1] if variants else question.strip()


def _question_sources(
    notebook: ResearchNotebook, attempts: list[RetrievalAttempt]
) -> list[EvidenceSource]:
    bundle_ids = {attempt.get("bundle_id") for attempt in attempts}
    return [
        source
        for bundle in notebook.bundles
        if bundle.bundle_id in bundle_ids
        for source in bundle.sources
        if source.evidence and source.content_origin != "search_snippet"
    ]


def _question_candidates(
    notebook: ResearchNotebook, item: QuestionPlan, year: int
) -> list[tuple[SuggestionKind, SearchPurpose, str, str]]:
    question = item["question"]
    core = _core_query(question)
    attempts = question_attempts(notebook, item["question_id"], question)
    sources = _question_sources(notebook, attempts)
    review = item["review"]
    candidates: list[tuple[SuggestionKind, SearchPurpose, str, str]] = [
        (
            "host_follow_up",
            "explore",
            query,
            "Your review listed this follow-up for an open gap.",
        )
        for query in (review["follow_up_queries"] if review else [])
    ]
    if not sources:
        candidates.append(
            (
                "initial",
                "explore",
                core,
                "No source has been read for this question yet."
                if not attempts
                else "Earlier retrievals for this question returned no readable "
                "source; search for an accessible full-text page.",
            )
        )
    elif not any(
        source_kind(source, notebook.preferred_domains) not in BACKGROUND_KINDS
        for source in sources
    ):
        if notebook.preferred_domains:
            query = f"{core} site:{notebook.preferred_domains[0]}"
        elif notebook.source_policy == "academic":
            query = f"{core} original paper"
        else:
            query = f"{core} official source"
        candidates.append(
            (
                "primary_source",
                "explore",
                query,
                "Only background sources (encyclopedia, community or metadata) "
                "support this question so far; find the original publisher or "
                "primary record.",
            )
        )
    domains = {_domain(source.url) for source in sources if _domain(source.url)}
    clusters = {source.content_cluster_id or source.content_hash for source in sources}
    if len(domains) == 1 or (sources and len(clusters) == 1):
        domain = next(iter(domains)) if len(domains) == 1 else ""
        candidates.append(
            (
                "independent_source",
                "explore",
                f"{core} -site:{domain}" if domain else core,
                (
                    f"All evidence so far comes from {domain}; "
                    if domain
                    else "All evidence so far is one content cluster; "
                )
                + "find an independent source (copies of the same text do not count).",
            )
        )
    if not any(
        attempt.get("purpose") == "counterevidence"
        and attempt.get("substantive_sources", 0) > 0
        for attempt in attempts
    ):
        candidates.append(
            (
                "counterevidence",
                "counterevidence",
                f"{core} limitations criticism",
                "No counterevidence has been read for this question; look for "
                "limitations, failures or sources that disagree.",
            )
        )
    for support in item["claim_support"]:
        if "single_source" in support["flags"]:
            claim = next(
                claim
                for claim in (review or {}).get("claims", [])
                if claim["claim_id"] == support["claim_id"]
            )
            excluded = " ".join(f"-site:{domain}" for domain in support["domains"])
            candidates.append(
                (
                    "corroboration",
                    "explore",
                    " ".join([keyword_query(claim["text"]), excluded]).strip(),
                    f"{support['claim_id']} rests on a single source; find an "
                    "independent source that states it.",
                )
            )
    asks_recent = _RECENCY_RE.search(question) or any(
        int(value) >= year for value in _YEAR_RE.findall(question)
    )
    if asks_recent and not any(
        (source.published_at or "").startswith(str(year)) for source in sources
    ):
        candidates.append(
            (
                "recency",
                "explore",
                f"{core} {year}" if str(year) not in core else core,
                f"The question asks about current information and no source read "
                f"so far is dated {year}; prefer recently published sources.",
            )
        )
    return candidates


def suggest_searches(
    notebook: ResearchNotebook, items: list[QuestionPlan]
) -> list[SuggestedQuery]:
    """Ranked search suggestions, at most three per open question."""
    year = datetime.now(timezone.utc).year
    suggestions: list[SuggestedQuery] = []
    for item in items:
        if not item["gaps"]:
            continue
        completed = {
            _normalized(attempt["query"])
            for attempt in question_attempts(
                notebook, item["question_id"], item["question"]
            )
            if attempt["status"] in {"ok", "insufficient_evidence"}
            and not attempt.get("url")
        }
        added = 0
        for kind, purpose, query, reason in _question_candidates(notebook, item, year):
            normalized = _normalized(query)
            if not normalized or normalized in completed or added >= PER_QUESTION:
                continue
            completed.add(normalized)
            added += 1
            suggestions.append(
                dict(
                    question_id=item["question_id"],
                    purpose=purpose,
                    kind=kind,
                    query=query,
                    reason=reason,
                )
            )
    return suggestions[:TOTAL]
