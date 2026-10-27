"""Deterministic assessment of the sources behind each factual claim.

For every supported or disputed fact, Sibyl reports how many sources the claim
cites, their source-kind hints and domains, and flags weak support: only
background sources, or a single source. Reviews already reject links to two
copies of one text, so each cited source is a distinct content cluster. The
claim checks then compare the fact with its cited text (see claim_checks).
These are routing heuristics over URLs and strings, not credibility verdicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .claim_checks import possible_conflict, unmatched_names, unquoted_numbers
from .dedup import canonical_url
from .evidence import EvidencePassage, EvidenceSource
from .source_policy import source_kind

if TYPE_CHECKING:
    from .job_models import EvidenceReview, RetrievalAttempt
    from .research_state import ResearchNotebook

BACKGROUND_KINDS = frozenset({"encyclopedia", "community", "scholarly_metadata"})
MAX_CONFLICTS = 5
PassageIndex = dict[str, tuple[EvidencePassage, EvidenceSource]]


def _passage_index(notebook: ResearchNotebook) -> PassageIndex:
    return {
        passage.citation_id: (passage, source)
        for bundle in notebook.bundles
        for source in bundle.sources
        if source.content_origin != "search_snippet"
        for passage in source.evidence
    }


def _facts(notebook: ResearchNotebook) -> list[dict[str, Any]]:
    return [
        claim
        for review in notebook.reviews.values()
        for claim in review.get("claims", [])
        if claim["kind"] == "fact" and claim["status"] in {"supported", "disputed"}
    ]


def claim_support(
    notebook: ResearchNotebook,
    claim: dict[str, Any],
    passages: PassageIndex | None = None,
    facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    passages = passages if passages is not None else _passage_index(notebook)
    facts = facts if facts is not None else _facts(notebook)
    sources: dict[str, EvidenceSource] = {}
    for passage_id in claim["supporting_passage_ids"]:
        if passage_id in passages:
            source = passages[passage_id][1]
            sources.setdefault(canonical_url(source.url), source)
    kinds = sorted(
        {source_kind(source, notebook.preferred_domains) for source in sources.values()}
    )
    domains = sorted(
        {
            (urlsplit(source.url).hostname or "").lower().removeprefix("www.")
            for source in sources.values()
        }
        - {""}
    )
    cited = [
        text
        for passage_id in [
            *claim["supporting_passage_ids"],
            *claim["challenging_passage_ids"],
        ]
        if passage_id in passages
        for text in (passages[passage_id][0].text, passages[passage_id][1].title)
    ]
    quotes = [quote["text"] for quote in claim.get("quotes", [])]
    checks = {
        # Claims stored before quotes were required are not re-validated.
        "unquoted_numbers": unquoted_numbers(claim["text"], quotes) if quotes else [],
        "unmatched_names": unmatched_names(claim["text"], cited) if cited else [],
        "conflicts_with": [
            other["claim_id"]
            for other in facts
            if claim["status"] == other["status"] == "supported"
            and other["claim_id"] != claim["claim_id"]
            and possible_conflict(claim["text"], other["text"])
        ][:MAX_CONFLICTS],
    }
    flags = []
    if sources and set(kinds) <= BACKGROUND_KINDS:
        flags.append("background_only")
    if len(sources) == 1:
        flags.append("single_source")
    for key, flag in (
        ("unquoted_numbers", "number_not_quoted"),
        ("unmatched_names", "name_not_in_sources"),
        ("conflicts_with", "possible_conflict"),
    ):
        if checks[key]:
            flags.append(flag)
    return {
        "claim_id": claim["claim_id"],
        "sources": len(sources),
        "source_kinds": kinds,
        "domains": domains,
        "flags": flags,
        **{key: value for key, value in checks.items() if value},
    }


def question_claim_support(
    notebook: ResearchNotebook, question_id: str
) -> list[dict[str, Any]]:
    """Support summaries for the question's supported and disputed facts."""
    review = notebook.reviews.get(question_id)
    if not review:
        return []
    passages, facts = _passage_index(notebook), _facts(notebook)
    return [
        claim_support(notebook, claim, passages, facts)
        for claim in review.get("claims", [])
        if claim["kind"] == "fact" and claim["status"] in {"supported", "disputed"}
    ]


def job_claim_support(notebook: ResearchNotebook) -> dict[str, dict[str, Any]]:
    """Support summaries for every supported or disputed fact, by claim ID."""
    passages, facts = _passage_index(notebook), _facts(notebook)
    return {
        claim["claim_id"]: claim_support(notebook, claim, passages, facts)
        for claim in facts
    }


def counterevidence_summary(
    notebook: ResearchNotebook,
    review: EvidenceReview | None,
    attempts: list[RetrievalAttempt],
) -> dict[str, Any]:
    """What the question's counterevidence reads produced and what the review links.

    `not_read`: no counterevidence read returned a substantive source.
    `linked`: the review cites a passage from such a read.
    `none_linked`: sources were read but none is cited; the review's
    counterevidence note is the host's account of why.
    """
    reads = [
        attempt
        for attempt in attempts
        if attempt.get("purpose") == "counterevidence"
        and attempt.get("substantive_sources", 0) > 0
        and attempt["status"] in {"ok", "insufficient_evidence"}
    ]
    bundle_ids = {attempt.get("bundle_id") for attempt in reads}
    read_passages = {
        passage.citation_id
        for bundle in notebook.bundles
        if bundle.bundle_id in bundle_ids
        for source in bundle.sources
        if source.content_origin != "search_snippet"
        for passage in source.evidence
    }
    cited = (
        {*review["supporting_passage_ids"], *review["challenging_passage_ids"]}
        if review
        else set()
    )
    linked = sorted(read_passages & cited)
    return {
        "status": "not_read" if not reads else ("linked" if linked else "none_linked"),
        "reads": len(reads),
        "linked_passage_ids": linked,
    }
