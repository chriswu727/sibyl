"""Deterministic assessment of the sources behind each factual claim.

For every supported or disputed fact, Sibyl reports how many sources the claim
cites, their source-kind hints and domains, and flags weak support: only
background sources, or a single source. Reviews already reject links to two
copies of one text, so each cited source is a distinct content cluster. These
are routing heuristics over URLs, not credibility verdicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .dedup import canonical_url
from .evidence import EvidenceSource
from .source_policy import source_kind

if TYPE_CHECKING:
    from .research_state import ResearchNotebook

BACKGROUND_KINDS = frozenset({"encyclopedia", "community", "scholarly_metadata"})


def _passage_sources(notebook: ResearchNotebook) -> dict[str, EvidenceSource]:
    return {
        passage.citation_id: source
        for bundle in notebook.bundles
        for source in bundle.sources
        if source.content_origin != "search_snippet"
        for passage in source.evidence
    }


def claim_support(
    notebook: ResearchNotebook,
    claim: dict[str, Any],
    passage_sources: dict[str, EvidenceSource] | None = None,
) -> dict[str, Any]:
    lookup = (
        passage_sources if passage_sources is not None else _passage_sources(notebook)
    )
    sources: dict[str, EvidenceSource] = {}
    for passage_id in claim["supporting_passage_ids"]:
        source = lookup.get(passage_id)
        if source is not None:
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
    flags = []
    if sources and set(kinds) <= BACKGROUND_KINDS:
        flags.append("background_only")
    if len(sources) == 1:
        flags.append("single_source")
    return {
        "claim_id": claim["claim_id"],
        "sources": len(sources),
        "source_kinds": kinds,
        "domains": domains,
        "flags": flags,
    }


def question_claim_support(
    notebook: ResearchNotebook, question_id: str
) -> list[dict[str, Any]]:
    """Support summaries for the question's supported and disputed facts."""
    review = notebook.reviews.get(question_id)
    if not review:
        return []
    lookup = _passage_sources(notebook)
    return [
        claim_support(notebook, claim, lookup)
        for claim in review.get("claims", [])
        if claim["kind"] == "fact" and claim["status"] in {"supported", "disputed"}
    ]
