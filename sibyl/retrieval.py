"""Keyless retrieval pipeline producing structured source bundles."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, cast
from urllib.parse import urlsplit

import httpx

from .content_clusters import cluster_content
from .context import relevant_window_span
from .dedup import canonical_url, dedup_pages
from .evidence import (
    BundleDiagnostics,
    BundleStatus,
    EvidenceSufficiency,
    EvidencePassage,
    EvidenceSource,
    QueryComplexity,
    RecommendedAction,
    SourceBundle,
)
from .passages import TextPassage, split_passages
from .queries import query_requires_decomposition, search_query_variants
from .ranking import (
    RankingBackend,
    flashrank_relevance_scores,
    lexical_query_coverage,
    lexical_query_terms,
    lexical_relevance_scores,
)
from .scraper import WebPage, scrape_urls
from .search import (
    fetch_wikipedia_extract,
    search_web,
    search_wikipedia,
    wikipedia_lookup,
)


_ACADEMIC_QUERY_RE = re.compile(
    r"\b(?:abstract|citation|doi|journal|paper|preprint|publication|study)\b",
    re.IGNORECASE,
)
_QUERY_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_QUOTED_TEXT_RE = re.compile(
    r'''["“]([^"”]+)["”]|(?<!\w)'([^']{3,})'(?!\w)'''
)
_QUESTION_OPENERS = {
    "after", "at", "how", "in", "on", "the", "what", "when", "where", "which", "who",
}
_FUTURE_YEAR_RE = re.compile(r"\b[2-9]\d{3}\b")


def _query_anchor_terms(query: str) -> set[str]:
    anchors = set()
    words = _QUERY_WORD_RE.findall(query)
    for index, word in enumerate(words):
        if len(word) < 4 or not word[0].isupper():
            continue
        if index == 0 and word.casefold() in _QUESTION_OPENERS:
            continue
        anchors.update(lexical_query_terms(word))
    for match in _QUOTED_TEXT_RE.finditer(query):
        anchors.update(lexical_query_terms(match.group(1) or match.group(2)))
    return anchors


def _entity_lookup_queries(query: str) -> List[str]:
    quoted = []
    for match in _QUOTED_TEXT_RE.finditer(query):
        value = " ".join((match.group(1) or match.group(2)).split())
        if value:
            quoted.append(value)
    if quoted:
        return quoted[:2]

    sequences = []
    current = []
    for index, word in enumerate(_QUERY_WORD_RE.findall(query)):
        capitalized = len(word) >= 2 and word[0].isupper()
        is_opener = index == 0 and word.casefold() in _QUESTION_OPENERS
        if capitalized and not is_opener:
            current.append(word)
        elif current:
            sequences.append(" ".join(current))
            current = []
    if current:
        sequences.append(" ".join(current))
    return sequences[:2]


def _future_outcome_is_unobservable(query: str) -> bool:
    if " will " not in f" {query.casefold()} ":
        return False
    current_year = datetime.now(timezone.utc).year
    return any(int(year) > current_year for year in _FUTURE_YEAR_RE.findall(query))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bundle_id(
    query: str, status: str, source_fingerprints: List[Tuple[str, str, str]]
) -> str:
    payload = json.dumps(
        {"query": query.strip(), "status": status, "sources": source_fingerprints},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"sb_{_sha256(payload)[:16]}"


def _source_type(url: str, result_types: Dict[str, str]) -> str:
    if "wikipedia.org/wiki/" in url:
        return "wikipedia"
    return result_types.get(canonical_url(url), "web")


def _assess_evidence_sufficiency(
    *,
    source_count: int,
    substantive_sources: int,
    evidence_chars: int,
    query_terms: int,
    query_term_coverage: float,
    unique_domains: int,
    independent_content_clusters: int,
    missing_anchor_terms: List[str],
    future_outcome_unobservable: bool,
    fragmented_query_support: bool,
    multi_step_query: bool,
) -> Tuple[EvidenceSufficiency, List[str]]:
    if source_count == 0:
        return "insufficient", ["no_sources"]

    blockers = []
    if substantive_sources == 0:
        blockers.append("no_substantive_sources")
    if evidence_chars < 200:
        blockers.append("too_little_evidence_text")
    if query_terms > 0 and query_term_coverage < 0.25:
        blockers.append("low_query_term_coverage")
    if missing_anchor_terms:
        blockers.append("missing_query_anchor")
    if future_outcome_unobservable:
        blockers.append("future_outcome_not_observable")
    if fragmented_query_support:
        blockers.append("fragmented_query_support")
    if multi_step_query:
        blockers.append("multi_step_query")
    if blockers:
        return "insufficient", blockers

    limitations = []
    if substantive_sources < 2:
        limitations.append("fewer_than_two_substantive_sources")
    elif independent_content_clusters < 2:
        limitations.append("fewer_than_two_independent_contents")
    if unique_domains < 2:
        limitations.append("single_domain")
    if query_terms == 0:
        limitations.append("query_has_no_lexical_terms")
    if limitations:
        return "limited", limitations
    return "sufficient", []


_SUFFICIENCY_REASON_LABELS = {
    "no_sources": "no sources",
    "no_substantive_sources": "no substantive full-text sources",
    "too_little_evidence_text": "too little evidence text",
    "low_query_term_coverage": "low query-term coverage",
    "missing_query_anchor": "one or more key query entities are absent",
    "future_outcome_not_observable": "the requested future outcome is not yet observable",
    "fragmented_query_support": (
        "no single source sufficiently covers the quoted target and requested fact"
    ),
    "multi_step_query": "the question contains a dependent fact chain",
}


def _insufficient_evidence_error(
    query: str,
    source_count: int,
    reasons: List[str],
) -> str:
    if source_count == 0:
        return f"No sources found for query: {query!r}. Try a different phrasing."
    details = ", ".join(
        _SUFFICIENCY_REASON_LABELS.get(reason, reason.replace("_", " "))
        for reason in reasons
    )
    return (
        f"Found {source_count} source(s), but the evidence is insufficient for "
        f"synthesis ({details}). Treat these as leads and gather more evidence."
    )


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
        scores = await asyncio.to_thread(
            flashrank_relevance_scores, query, documents
        )
        return scores, "flashrank", ""
    except Exception as exc:
        detail = str(exc).strip().replace("\n", " ")[:160]
        reason = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
        warning = f"FlashRank failed ({reason}); fell back to lexical_v1."
        return lexical_relevance_scores(query, documents), "lexical_v1", warning


async def gather_source_bundle(
    query: str,
    max_sources: int = 10,
    chars_per_source: int = 7000,
    client: Optional[httpx.AsyncClient] = None,
    ranker: RankingBackend = "lexical",
    render_thin_pages: bool = False,
) -> SourceBundle:
    started_at = time.monotonic()
    requested_ranker = str(ranker or "").strip().lower()
    try:
        requested_max_sources = int(max_sources)
        requested_chars_per_source = int(chars_per_source)
    except (TypeError, ValueError):
        requested_max_sources = 10
        requested_chars_per_source = 7000
        diagnostics = _diagnostics(
            requested_max_sources=requested_max_sources,
            effective_max_sources=10,
            requested_chars_per_source=requested_chars_per_source,
            effective_chars_per_source=7000,
            started_at=started_at,
            requested_ranking_method=requested_ranker,
            recommended_action="revise_request",
        )
        return SourceBundle(
            schema_version="1.6",
            bundle_id=_bundle_id(str(query or ""), "invalid_request", []),
            query=str(query or ""),
            status="invalid_request",
            sources=[],
            diagnostics=diagnostics,
            error="max_sources and chars_per_source must be integers.",
        )

    effective_max_sources = max(1, min(20, requested_max_sources))
    effective_chars_per_source = max(500, min(10000, requested_chars_per_source))
    url_attempt_limit = max(effective_max_sources + 2, 12)
    clean_query = str(query or "").strip()
    search_queries = search_query_variants(clean_query)
    multi_step_query = query_requires_decomposition(clean_query)
    query_complexity: QueryComplexity = (
        "multi_step" if multi_step_query else "single_step"
    )
    if requested_ranker not in {"lexical", "flashrank", "none"}:
        diagnostics = _diagnostics(
            requested_max_sources=requested_max_sources,
            effective_max_sources=effective_max_sources,
            requested_chars_per_source=requested_chars_per_source,
            effective_chars_per_source=effective_chars_per_source,
            started_at=started_at,
            requested_ranking_method=requested_ranker,
            query_complexity=query_complexity,
            recommended_action="revise_request",
        )
        return SourceBundle(
            schema_version="1.6",
            bundle_id=_bundle_id(clean_query, "invalid_request", []),
            query=clean_query,
            status="invalid_request",
            sources=[],
            diagnostics=diagnostics,
            error="ranker must be one of: lexical, flashrank, none.",
        )
    selected_ranker = cast(RankingBackend, requested_ranker)
    if not clean_query:
        diagnostics = _diagnostics(
            requested_max_sources=requested_max_sources,
            effective_max_sources=effective_max_sources,
            requested_chars_per_source=requested_chars_per_source,
            effective_chars_per_source=effective_chars_per_source,
            started_at=started_at,
            requested_ranking_method=requested_ranker,
            query_complexity="not_assessed",
            recommended_action="revise_request",
        )
        return SourceBundle(
            schema_version="1.6",
            bundle_id=_bundle_id(clean_query, "invalid_request", []),
            query=clean_query,
            status="invalid_request",
            sources=[],
            diagnostics=diagnostics,
            error="query must not be empty.",
        )

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=12.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    results = []
    urls: List[str] = []
    pages: List[WebPage] = []
    snippet_fallbacks = 0
    wikipedia_fallbacks = 0
    metadata_fallbacks = 0
    candidates: List[WebPage] = []
    try:
        search_batches = []
        include_academic = bool(_ACADEMIC_QUERY_RE.search(clean_query))
        for search_query in search_queries:
            search_batches.append(
                await search_web(
                    search_query,
                    "all",
                    max_results=6,
                    client=client,
                    include_academic=include_academic,
                )
            )
        entity_batches = await asyncio.gather(
            *[
                search_wikipedia(entity_query, 2, client=client)
                for entity_query in _entity_lookup_queries(clean_query)
                if entity_query.casefold()
                not in {query.casefold() for query in search_queries}
            ]
        )
        search_batches = [*entity_batches, *search_batches]
        seen_results = set()
        for batch in search_batches:
            for result in batch:
                result_key = canonical_url(result.url)
                if result_key not in seen_results:
                    seen_results.add(result_key)
                    results.append(result)
        seen = set()
        for result in results:
            if result.url.startswith("http") and result.url not in seen:
                seen.add(result.url)
                urls.append(result.url)

        attempted_urls = urls[:url_attempt_limit]
        wikipedia_results = []
        seen_wikipedia_urls = set()
        for result in results:
            if "wikipedia.org/wiki/" not in result.url:
                continue
            key = canonical_url(result.url)
            if key in seen_wikipedia_urls:
                continue
            seen_wikipedia_urls.add(key)
            wikipedia_results.append(result)
            if len(wikipedia_results) == 3:
                break
        direct_pages, wikipedia_extracts = await asyncio.gather(
            scrape_urls(
                attempted_urls,
                max_chars=30000,
                concurrency=12,
                client=client,
                js_render=render_thin_pages,
            ),
            asyncio.gather(
                *[
                    fetch_wikipedia_extract(result.url, client)
                    for result in wikipedia_results
                ],
                return_exceptions=True,
            ),
        )
        pages = direct_pages
        good = [page for page in pages if page.text and len(page.text) > 150 and not page.error]
        for result, extract in zip(wikipedia_results, wikipedia_extracts):
            if isinstance(extract, str) and extract:
                good.append(
                    WebPage(
                        url=result.url,
                        title=result.title,
                        text=extract,
                        content_origin="wikipedia_api",
                    )
                )
                wikipedia_fallbacks += 1
        scraped = {page.url for page in good}
        for result in results:
            if result.url not in scraped and result.snippet and len(result.snippet) > 120:
                content_origin = (
                    "crossref_api"
                    if result.provider == "crossref"
                    else "search_snippet"
                )
                good.append(
                    WebPage(
                        url=result.url,
                        title=result.title,
                        text=result.snippet,
                        content_origin=content_origin,
                    )
                )
                if content_origin == "crossref_api":
                    metadata_fallbacks += 1
                else:
                    snippet_fallbacks += 1

        good = dedup_pages(good)
        substantive = [page for page in good if len(page.text) > 200]
        if len(substantive) < 3:
            wiki_pages = await wikipedia_lookup(
                search_queries[-1], client=client, max_pages=2
            )
            wikipedia_fallbacks += len(wiki_pages)
            if wiki_pages:
                good = dedup_pages(good + wiki_pages)
                substantive = [page for page in good if len(page.text) > 200]
        candidates = good

    except Exception as exc:
        diagnostics = _diagnostics(
            search_results=len(results),
            unique_urls=len(urls),
            urls_attempted=min(len(urls), url_attempt_limit),
            pages_scraped=len(pages),
            scrape_failures=sum(1 for page in pages if page.error),
            snippet_fallbacks=snippet_fallbacks,
            wikipedia_fallbacks=wikipedia_fallbacks,
            metadata_fallbacks=metadata_fallbacks,
            requested_max_sources=requested_max_sources,
            effective_max_sources=effective_max_sources,
            requested_chars_per_source=requested_chars_per_source,
            effective_chars_per_source=effective_chars_per_source,
            started_at=started_at,
            requested_ranking_method=requested_ranker,
            search_queries=search_queries,
            search_providers=sorted(
                {result.provider for result in results if result.provider}
            ),
            query_complexity=query_complexity,
            recommended_action="retry",
        )
        return SourceBundle(
            schema_version="1.6",
            bundle_id=_bundle_id(clean_query, "failed", []),
            query=clean_query,
            status="failed",
            sources=[],
            diagnostics=diagnostics,
            error=f"Retrieval failed: {str(exc)[:200]}",
        )
    finally:
        if own_client and client is not None:
            await client.aclose()

    result_types = {}
    result_titles = {}
    for result in results:
        key = canonical_url(result.url)
        result_types.setdefault(key, result.source)
        result_titles.setdefault(key, result.title)

    content_clusters = cluster_content([page.text for page in candidates])
    cluster_id_by_source_hash = {}

    source_candidates = []
    ranking_documents = []
    for page, cluster_id in zip(candidates, content_clusters.cluster_ids):
        representative_start, representative_end = relevant_window_span(
            clean_query, page.text, width=effective_chars_per_source
        )
        representative = page.text[representative_start:representative_end]
        page_key = canonical_url(page.url)
        title = page.title or result_titles.get(page_key, page.url)
        source_hash = _sha256(page.text)
        cluster_id_by_source_hash[source_hash] = cluster_id
        source_candidates.append(
            (page, title, source_hash, representative_start, representative)
        )
        ranking_documents.append((title, representative))

    relevance_scores, source_ranking_method, source_ranking_warning = (
        await _score_documents(clean_query, ranking_documents, selected_ranker)
    )
    ranked_indices = sorted(
        range(len(source_candidates)),
        key=lambda index: (-(relevance_scores[index] or 0.0), index),
    )
    if selected_ranker == "none":
        selected_indices = ranked_indices[:effective_max_sources]
    else:
        diverse_indices = []
        duplicate_indices = []
        seen_clusters = set()
        for index in ranked_indices:
            cluster_id = content_clusters.cluster_ids[index]
            if cluster_id in seen_clusters:
                duplicate_indices.append(index)
            else:
                diverse_indices.append(index)
                seen_clusters.add(cluster_id)
        selected_indices = (diverse_indices + duplicate_indices)[
            :effective_max_sources
        ]
    selected_sources = [
        (*source_candidates[index], relevance_scores[index])
        for index in selected_indices
    ]

    passage_size = min(2500, max(500, effective_chars_per_source // 3))
    max_passages_per_source = min(
        3, max(1, effective_chars_per_source // passage_size)
    )
    chunk_groups = []
    chunk_documents = []
    for (
        page, title, source_hash, representative_start, representative, relevance_score
    ) in selected_sources:
        chunks = split_passages(
            representative, max_chars=passage_size, overlap_chars=0
        )
        if len(chunks) > max_passages_per_source:
            tail_start = chunks[max_passages_per_source - 1].start_char
            chunks = chunks[:max_passages_per_source - 1] + [
                TextPassage(
                    representative[tail_start:],
                    tail_start,
                    len(representative),
                )
            ]
        score_start = len(chunk_documents)
        chunk_documents.extend((title, chunk.text) for chunk in chunks)
        chunk_groups.append(
            (
                page,
                title,
                source_hash,
                representative_start,
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
    passage_scores, passage_ranking_method, passage_ranking_warning = (
        await _score_documents(clean_query, chunk_documents, passage_ranker)
    )
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
        representative_start,
        relevance_score,
        chunks,
        score_start,
    ) in chunk_groups:
        chunk_scores = passage_scores[score_start:score_start + len(chunks)]
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
                    TextPassage(
                        chunk.text,
                        representative_start + chunk.start_char,
                        representative_start + chunk.end_char,
                    ),
                    content_hash,
                    chunk_scores[chunk_index],
                )
            )
            seen_hashes.add(content_hash)
        prepared.append(
            (page, title, source_hash, relevance_score, selected_passages)
        )

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
    evidence_terms = set().union(
        *(lexical_query_terms(text) for text in evidence_texts)
    ) if evidence_texts else set()
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
            len(page.text) > 200
            and page.content_origin != "search_snippet"
            and (not query_anchor_terms or query_anchor_terms <= source_terms)
        )
    ]
    substantive_sources = len(substantive_prepared)
    selected_cluster_ids = {
        cluster_id_by_source_hash[source_hash]
        for _, _, source_hash, _, _ in prepared
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
        unique_domains=len(domains),
        independent_content_clusters=independent_content_clusters,
        missing_anchor_terms=missing_anchor_terms,
        future_outcome_unobservable=_future_outcome_is_unobservable(clean_query),
        fragmented_query_support=(
            bool(_QUOTED_TEXT_RE.search(clean_query))
            and max_source_query_term_coverage < 0.6
        ),
        multi_step_query=multi_step_query,
    )
    recommended_action: RecommendedAction
    if multi_step_query:
        recommended_action = "decompose_query"
    elif evidence_sufficiency == "sufficient":
        recommended_action = "synthesize"
    else:
        recommended_action = "refine_query"
    bundle_status: BundleStatus = (
        "ok" if evidence_sufficiency == "sufficient" else "insufficient_evidence"
    )
    bundle_id = _bundle_id(clean_query, bundle_status, fingerprints)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    sources = []
    for index, (
        page, title, source_hash, relevance_score, selected_passages
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
                source_type=_source_type(page.url, result_types),
                char_count=len(page.text),
                evidence=evidence,
                content_origin=page.content_origin,
                published_at=page.published_at,
                published_at_method=page.published_at_method,
                content_cluster_id=cluster_id_by_source_hash[source_hash],
                relevance_score=relevance_score,
            )
        )

    attempted_count = min(len(urls), url_attempt_limit)
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
        candidates_ranked=(
            len(source_candidates) if selected_ranker != "none" else 0
        ),
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


def render_source_bundle(bundle: SourceBundle) -> str:
    if bundle.status == "invalid_request":
        return f"Invalid retrieval request: {bundle.error}"
    if bundle.status == "failed":
        return bundle.error or "Retrieval failed."
    if not bundle.sources:
        return (
            bundle.error
            or f"No sources found for query: {bundle.query!r}. Try a different phrasing."
        )

    parts = []
    for index, source in enumerate(bundle.sources, 1):
        text = "\n\n".join(passage.text for passage in source.evidence)
        parts.append(f"[Source {index}: {source.title}]\nURL: {source.url}\n{text}\n")
    rendered = (
        f"Retrieved {len(bundle.sources)} sources for query {bundle.query!r}. "
        "Reason over these and "
        f"cite [Source N]; if the answer isn't here, gather more or say you don't know.\n\n"
        + "\n---\n".join(parts)
    )
    if bundle.status == "insufficient_evidence":
        return f"Evidence warning: {bundle.error}\n\n{rendered}"
    return rendered
