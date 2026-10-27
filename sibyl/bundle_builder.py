"""Turn candidate pages into a ranked, deduplicated and assessed SourceBundle.

Discovery (Sibyl's keyless search) and host-supplied source reads both end here,
so every source receives the same passage selection, clustering, ranking and
sufficiency assessment.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

from .content_clusters import cluster_content
from .dedup import canonical_url
from .evidence import (
    BundleDiagnostics,
    BundleStatus,
    EvidencePassage,
    EvidenceSource,
    EvidenceSufficiency,
    QueryComplexity,
    RecommendedAction,
    SourceBundle,
)
from .passages import select_passages
from .ranking import (
    RankingBackend,
    flashrank_relevance_scores,
    lexical_query_coverage,
    lexical_query_terms,
    lexical_relevance_scores,
)
from .retrieval_policy import (
    _QUOTED_TEXT_RE,
    _assess_evidence_sufficiency,
    _bundle_id,
    _future_outcome_is_unobservable,
    _has_historical_role_support,
    _insufficient_evidence_error,
    _query_anchor_terms,
    _sha256,
    _source_type,
)
from .scraper import WebPage
from .search import SearchResult
from .source_policy import matches_domain, source_kind
from .temporal import period_match


@dataclass
class Discovery:
    """How the candidate pages were found; empty for host-supplied sources."""

    results: List[SearchResult] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    attempted_urls: List[str] = field(default_factory=list)
    pages: List[WebPage] = field(default_factory=list)
    snippet_fallbacks: int = 0
    wikipedia_fallbacks: int = 0
    metadata_fallbacks: int = 0
    search_queries: List[str] = field(default_factory=list)
    refinement_searches: int = 0
    refinement_failures: int = 0


def _diagnostics(
    *,
    search_results: int = 0,
    unique_urls: int = 0,
    urls_attempted: int = 0,
    pages_scraped: int = 0,
    scrape_failures: int = 0,
    snippet_fallbacks: int = 0,
    wikipedia_fallbacks: int = 0,
    metadata_fallbacks: int = 0,
    sources_returned: int = 0,
    requested_max_sources: int,
    effective_max_sources: int,
    requested_chars_per_source: int,
    effective_chars_per_source: int,
    started_at: float,
    requested_ranking_method: str = "lexical",
    ranking_method: str = "not_run",
    ranking_warning: str = "",
    candidates_ranked: int = 0,
    chunks_ranked: int = 0,
    passages_returned: int = 0,
    passage_size: int = 0,
    max_passages_per_source: int = 0,
    coverage_method: str = "not_run",
    query_terms: int = 0,
    matched_query_terms: int = 0,
    query_term_coverage: Optional[float] = None,
    unique_domains: int = 0,
    substantive_sources: int = 0,
    evidence_chars: int = 0,
    candidate_content_clusters: int = 0,
    duplicate_candidates: int = 0,
    independent_content_clusters: int = 0,
    duplicate_sources: int = 0,
    content_cluster_method: str = "not_run",
    evidence_sufficiency: EvidenceSufficiency = "not_assessed",
    sufficiency_reasons: Optional[List[str]] = None,
    search_queries: Optional[List[str]] = None,
    search_providers: Optional[List[str]] = None,
    max_source_query_term_coverage: Optional[float] = None,
    query_complexity: QueryComplexity = "not_assessed",
    recommended_action: RecommendedAction = "not_assessed",
    refinement_searches: int = 0,
    refinement_failures: int = 0,
    substantive_domains: int = 0,
) -> BundleDiagnostics:
    return BundleDiagnostics(
        search_results=search_results,
        unique_urls=unique_urls,
        urls_attempted=urls_attempted,
        pages_scraped=pages_scraped,
        scrape_failures=scrape_failures,
        snippet_fallbacks=snippet_fallbacks,
        wikipedia_fallbacks=wikipedia_fallbacks,
        sources_returned=sources_returned,
        requested_max_sources=requested_max_sources,
        effective_max_sources=effective_max_sources,
        requested_chars_per_source=requested_chars_per_source,
        effective_chars_per_source=effective_chars_per_source,
        latency_ms=round((time.monotonic() - started_at) * 1000),
        requested_ranking_method=requested_ranking_method,
        ranking_method=ranking_method,
        ranking_warning=ranking_warning,
        candidates_ranked=candidates_ranked,
        chunks_ranked=chunks_ranked,
        passages_returned=passages_returned,
        passage_size=passage_size,
        max_passages_per_source=max_passages_per_source,
        coverage_method=coverage_method,
        query_terms=query_terms,
        matched_query_terms=matched_query_terms,
        query_term_coverage=query_term_coverage,
        unique_domains=unique_domains,
        substantive_sources=substantive_sources,
        evidence_chars=evidence_chars,
        candidate_content_clusters=candidate_content_clusters,
        duplicate_candidates=duplicate_candidates,
        independent_content_clusters=independent_content_clusters,
        duplicate_sources=duplicate_sources,
        content_cluster_method=content_cluster_method,
        evidence_sufficiency=evidence_sufficiency,
        sufficiency_reasons=sufficiency_reasons or [],
        search_queries=search_queries or [],
        search_providers=search_providers or [],
        max_source_query_term_coverage=max_source_query_term_coverage,
        metadata_fallbacks=metadata_fallbacks,
        query_complexity=query_complexity,
        recommended_action=recommended_action,
        refinement_searches=refinement_searches,
        refinement_failures=refinement_failures,
        substantive_domains=substantive_domains,
    )


async def _score_documents(
    query: str,
    documents: List[Tuple[str, str]],
    ranker: RankingBackend,
) -> Tuple[List[Optional[float]], str, str]:
    if ranker == "none":
        return [None] * len(documents), "none", ""
    if ranker == "lexical":
        return lexical_relevance_scores(query, documents), "lexical_v1", ""

    try:
        scores = await asyncio.to_thread(flashrank_relevance_scores, query, documents)
        return scores, "flashrank", ""
    except Exception as exc:
        reason = type(exc).__name__
        warning = f"FlashRank failed ({reason}); fell back to lexical_v1."
        return lexical_relevance_scores(query, documents), "lexical_v1", warning


def _has_substantive_text(page: WebPage) -> bool:
    return (
        len(page.text) > 200
        and not page.error
        and page.content_origin not in {"search_snippet", "crossref_api"}
    )


async def build_bundle(
    clean_query: str,
    candidates: List[WebPage],
    discovery: Discovery,
    *,
    requested_max_sources: int,
    effective_max_sources: int,
    requested_chars_per_source: int,
    effective_chars_per_source: int,
    requested_ranker: str,
    selected_ranker: RankingBackend,
    preferred_hosts: List[str],
    multi_step_query: bool,
    started_at: float,
    max_passages_per_source: int | None = None,
    status_from_read: bool = False,
) -> SourceBundle:
    """Assemble a SourceBundle from candidate pages.

    Discovery passes status_from_read=False: the bundle is "ok" only when the
    evidence is sufficient. Host-supplied reads pass True: a read that yields
    sources is "ok" (it succeeded), while evidence_sufficiency still reports how
    much the sources support the query.
    """
    results = discovery.results
    urls = discovery.urls
    attempted_urls = discovery.attempted_urls
    pages = discovery.pages
    snippet_fallbacks = discovery.snippet_fallbacks
    wikipedia_fallbacks = discovery.wikipedia_fallbacks
    metadata_fallbacks = discovery.metadata_fallbacks
    search_queries = discovery.search_queries
    refinement_searches = discovery.refinement_searches
    refinement_failures = discovery.refinement_failures
    query_complexity: QueryComplexity = (
        "multi_step" if multi_step_query else "single_step"
    )
    result_types = {}
    result_titles = {}
    for result in results:
        key = canonical_url(result.url)
        result_types.setdefault(key, result.source)
        result_titles.setdefault(key, result.title)

    content_clusters = cluster_content([page.text for page in candidates])
    cluster_id_by_source_hash = {}
    passage_size = min(2500, max(500, effective_chars_per_source // 3))
    if max_passages_per_source is None:
        max_passages_per_source = min(
            3, max(1, effective_chars_per_source // passage_size)
        )

    source_candidates = []
    ranking_documents = []
    for page, cluster_id in zip(candidates, content_clusters.cluster_ids):
        chunks = select_passages(
            clean_query,
            page.text,
            max_chars=effective_chars_per_source,
            max_passages=max_passages_per_source,
        )
        representative = "\n\n".join(chunk.text for chunk in chunks)
        page_key = canonical_url(page.url)
        title = page.title or result_titles.get(page_key, page.url)
        source_hash = _sha256(page.text)
        cluster_id_by_source_hash[source_hash] = cluster_id
        source_candidates.append((page, title, source_hash, chunks))
        ranking_documents.append((title, representative))

    (
        relevance_scores,
        source_ranking_method,
        source_ranking_warning,
    ) = await _score_documents(clean_query, ranking_documents, selected_ranker)

    preferred_representatives: set[int] = set()
    preferred_order = sorted(
        range(len(source_candidates)),
        key=lambda index: (
            -period_match(
                clean_query,
                source_candidates[index][1],
                source_candidates[index][0].url,
                source_candidates[index][0].published_at,
            ),
            -(relevance_scores[index] or 0.0),
            index,
        ),
    )
    for domain in preferred_hosts:
        representative_index = next(
            (
                index
                for index in preferred_order
                if (relevance_scores[index] or 0.0) > 0
                and source_candidates[index][0].content_origin
                not in {"search_snippet", "crossref_api"}
                and source_kind(source_candidates[index][0], preferred_hosts)
                not in {"community", "encyclopedia"}
                and matches_domain(
                    (urlsplit(source_candidates[index][0].url).hostname or "").lower(),
                    [domain],
                )
            ),
            None,
        )
        if representative_index is not None:
            preferred_representatives.add(representative_index)

    def source_preference(index: int) -> int:
        return int(index in preferred_representatives)

    ranked_indices = sorted(
        range(len(source_candidates)),
        key=lambda index: (
            -period_match(
                clean_query,
                source_candidates[index][1],
                source_candidates[index][0].url,
                source_candidates[index][0].published_at,
            )
            if selected_ranker != "none"
            else 0,
            -source_preference(index) if selected_ranker != "none" else 0,
            -(relevance_scores[index] or 0.0),
            index,
        ),
    )
    if selected_ranker == "none":
        selected_indices = ranked_indices[:effective_max_sources]
    else:
        domain_diverse_indices = []
        same_domain_indices = []
        duplicate_indices = []
        seen_clusters = set()
        seen_domains = set()
        for index in ranked_indices:
            cluster_id = content_clusters.cluster_ids[index]
            if cluster_id in seen_clusters:
                duplicate_indices.append(index)
                continue
            seen_clusters.add(cluster_id)
            domain = urlsplit(source_candidates[index][0].url).netloc.casefold()
            domain = domain.removeprefix("www.")
            if domain and domain not in seen_domains:
                domain_diverse_indices.append(index)
                seen_domains.add(domain)
            else:
                same_domain_indices.append(index)
        selected_indices = sorted(
            domain_diverse_indices + same_domain_indices + duplicate_indices,
            key=lambda index: (
                -period_match(
                    clean_query,
                    source_candidates[index][1],
                    source_candidates[index][0].url,
                    source_candidates[index][0].published_at,
                ),
                -source_preference(index),
            ),
        )[:effective_max_sources]
    selected_sources = [
        (*source_candidates[index], relevance_scores[index])
        for index in selected_indices
    ]

    chunk_groups = []
    chunk_documents = []
    for page, title, source_hash, chunks, relevance_score in selected_sources:
        score_start = len(chunk_documents)
        chunk_documents.extend((title, chunk.text) for chunk in chunks)
        chunk_groups.append(
            (
                page,
                title,
                source_hash,
                relevance_score,
                chunks,
                score_start,
            )
        )

    passage_ranker: RankingBackend = (
        "flashrank" if source_ranking_method == "flashrank" else selected_ranker
    )
    if source_ranking_method == "lexical_v1":
        passage_ranker = "lexical"
    (
        passage_scores,
        passage_ranking_method,
        passage_ranking_warning,
    ) = await _score_documents(clean_query, chunk_documents, passage_ranker)
    if source_ranking_method == passage_ranking_method:
        ranking_method = source_ranking_method
    else:
        ranking_method = (
            f"{source_ranking_method}_sources+{passage_ranking_method}_passages"
        )
    warnings = list(
        dict.fromkeys(
            warning
            for warning in [source_ranking_warning, passage_ranking_warning]
            if warning
        )
    )
    ranking_warning = " ".join(warnings)

    prepared = []
    for (
        page,
        title,
        source_hash,
        relevance_score,
        chunks,
        score_start,
    ) in chunk_groups:
        chunk_scores = passage_scores[score_start : score_start + len(chunks)]
        chunk_indices = sorted(
            range(len(chunks)),
            key=lambda index: (-(chunk_scores[index] or 0.0), index),
        )
        selected_passages = []
        seen_hashes = set()
        for chunk_index in chunk_indices:
            chunk = chunks[chunk_index]
            content_hash = _sha256(chunk.text)
            if content_hash in seen_hashes:
                continue
            selected_passages.append(
                (
                    chunk,
                    content_hash,
                    chunk_scores[chunk_index],
                )
            )
            seen_hashes.add(content_hash)
        prepared.append((page, title, source_hash, relevance_score, selected_passages))

    fingerprints = []
    for page, _, source_hash, _, selected_passages in prepared:
        passage_fingerprint = _sha256(
            "|".join(content_hash for _, content_hash, _ in selected_passages)
        )
        fingerprints.append((canonical_url(page.url), source_hash, passage_fingerprint))

    evidence_texts = [
        text
        for _, title, _, _, selected_passages in prepared
        for text in [title, *(chunk.text for chunk, _, _ in selected_passages)]
    ]
    coverage = lexical_query_coverage(clean_query, evidence_texts)
    query_anchor_terms = _query_anchor_terms(clean_query)
    source_evidence_terms = [
        set().union(
            *(
                lexical_query_terms(text)
                for text in [
                    title,
                    *(chunk.text for chunk, _, _ in selected_passages),
                ]
            )
        )
        for _, title, _, _, selected_passages in prepared
    ]
    source_coverages = [
        lexical_query_coverage(
            clean_query,
            [title, *(chunk.text for chunk, _, _ in selected_passages)],
        ).score
        for _, title, _, _, selected_passages in prepared
    ]
    max_source_query_term_coverage = max(source_coverages, default=0.0)
    evidence_terms = (
        set().union(*(lexical_query_terms(text) for text in evidence_texts))
        if evidence_texts
        else set()
    )
    missing_anchor_terms = sorted(query_anchor_terms - evidence_terms)
    domains = {
        urlsplit(page.url).netloc.lower().removeprefix("www.")
        for page, _, _, _, _ in prepared
        if urlsplit(page.url).netloc
    }
    substantive_prepared = [
        (page, source_hash)
        for (page, _, source_hash, _, _), source_terms in zip(
            prepared, source_evidence_terms
        )
        if (
            _has_substantive_text(page)
            and (not query_anchor_terms or query_anchor_terms <= source_terms)
        )
    ]
    substantive_sources = len(substantive_prepared)
    substantive_domains = {
        urlsplit(page.url).netloc.lower().removeprefix("www.")
        for page, _ in substantive_prepared
        if urlsplit(page.url).netloc
    }
    selected_cluster_ids = {
        cluster_id_by_source_hash[source_hash] for _, _, source_hash, _, _ in prepared
    }
    independent_content_cluster_ids = {
        cluster_id_by_source_hash[source_hash]
        for _, source_hash in substantive_prepared
    }
    independent_content_clusters = len(independent_content_cluster_ids)
    duplicate_sources = len(prepared) - len(selected_cluster_ids)
    evidence_chars = sum(
        len(chunk.text)
        for _, _, _, _, selected_passages in prepared
        for chunk, _, _ in selected_passages
    )
    evidence_sufficiency, sufficiency_reasons = _assess_evidence_sufficiency(
        source_count=len(prepared),
        substantive_sources=substantive_sources,
        evidence_chars=evidence_chars,
        query_terms=coverage.query_terms,
        query_term_coverage=coverage.score,
        unique_domains=len(substantive_domains),
        independent_content_clusters=independent_content_clusters,
        missing_anchor_terms=missing_anchor_terms,
        future_outcome_unobservable=_future_outcome_is_unobservable(clean_query),
        fragmented_query_support=(
            bool(_QUOTED_TEXT_RE.search(clean_query))
            and max_source_query_term_coverage < 0.6
        ),
        multi_step_query=multi_step_query,
        historical_role_support_missing=not _has_historical_role_support(
            clean_query,
            [
                " ".join([title, chunk.text])
                for _, title, _, _, selected_passages in prepared
                for chunk, _, _ in selected_passages
            ],
        ),
    )
    sufficiency_reasons.extend(
        dict.fromkeys(
            warning for page, *_ in prepared for warning in page.extraction_warnings
        )
    )
    recommended_action: RecommendedAction
    if multi_step_query:
        recommended_action = "decompose_query"
    elif evidence_sufficiency == "sufficient":
        recommended_action = "synthesize"
    else:
        recommended_action = "refine_query"
    succeeded = (
        bool(prepared) if status_from_read else evidence_sufficiency == "sufficient"
    )
    bundle_status: BundleStatus = "ok" if succeeded else "insufficient_evidence"
    bundle_id = _bundle_id(clean_query, bundle_status, fingerprints)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    sources = []
    for index, (
        page,
        title,
        source_hash,
        relevance_score,
        selected_passages,
    ) in enumerate(prepared, 1):
        source_id = f"S{index}"
        evidence = []
        for passage_index, (chunk, content_hash, passage_score) in enumerate(
            selected_passages, 1
        ):
            passage_id = f"P{passage_index}"
            evidence.append(
                EvidencePassage(
                    passage_id=passage_id,
                    citation_id=f"{bundle_id}/{source_id}/{passage_id}",
                    text=chunk.text,
                    content_hash=content_hash,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    score=passage_score,
                )
            )
        sources.append(
            EvidenceSource(
                source_id=source_id,
                url=page.url,
                title=title,
                retrieved_at=retrieved_at,
                content_hash=source_hash,
                source_type="pdf"
                if page.source_type == "pdf"
                else _source_type(page.url, result_types),
                char_count=len(page.text),
                evidence=evidence,
                content_origin=page.content_origin,
                published_at=page.published_at,
                published_at_method=page.published_at_method,
                content_cluster_id=cluster_id_by_source_hash[source_hash],
                relevance_score=relevance_score,
            )
        )

    attempted_count = len(attempted_urls)
    diagnostics = _diagnostics(
        search_results=len(results),
        unique_urls=len(urls),
        urls_attempted=attempted_count,
        pages_scraped=len(pages),
        scrape_failures=sum(1 for page in pages if page.error),
        snippet_fallbacks=snippet_fallbacks,
        wikipedia_fallbacks=wikipedia_fallbacks,
        metadata_fallbacks=metadata_fallbacks,
        sources_returned=len(sources),
        requested_max_sources=requested_max_sources,
        effective_max_sources=effective_max_sources,
        requested_chars_per_source=requested_chars_per_source,
        effective_chars_per_source=effective_chars_per_source,
        started_at=started_at,
        requested_ranking_method=requested_ranker,
        ranking_method=ranking_method if source_candidates else "not_run",
        ranking_warning=ranking_warning,
        candidates_ranked=(len(source_candidates) if selected_ranker != "none" else 0),
        chunks_ranked=len(chunk_documents) if selected_ranker != "none" else 0,
        passages_returned=sum(len(source.evidence) for source in sources),
        passage_size=passage_size if source_candidates else 0,
        max_passages_per_source=max_passages_per_source if source_candidates else 0,
        coverage_method="lexical_query_terms_v1" if sources else "not_run",
        query_terms=coverage.query_terms,
        matched_query_terms=coverage.matched_terms,
        query_term_coverage=coverage.score if sources else None,
        unique_domains=len(domains),
        substantive_sources=substantive_sources,
        substantive_domains=len(substantive_domains),
        evidence_chars=evidence_chars,
        candidate_content_clusters=content_clusters.cluster_count,
        duplicate_candidates=content_clusters.duplicate_count,
        independent_content_clusters=independent_content_clusters,
        duplicate_sources=duplicate_sources,
        content_cluster_method=(content_clusters.method if candidates else "not_run"),
        evidence_sufficiency=evidence_sufficiency,
        sufficiency_reasons=sufficiency_reasons,
        search_queries=search_queries,
        search_providers=sorted(
            {result.provider for result in results if result.provider}
        ),
        max_source_query_term_coverage=(
            max_source_query_term_coverage if sources else None
        ),
        query_complexity=query_complexity,
        recommended_action=recommended_action,
        refinement_searches=refinement_searches,
        refinement_failures=refinement_failures,
    )
    return SourceBundle(
        schema_version="1.6",
        bundle_id=bundle_id,
        query=clean_query,
        status=bundle_status,
        sources=sources,
        diagnostics=diagnostics,
        error=(
            _insufficient_evidence_error(
                clean_query,
                len(sources),
                sufficiency_reasons,
            )
            if bundle_status == "insufficient_evidence"
            else ""
        ),
    )
