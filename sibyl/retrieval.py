"""Keyless retrieval pipeline producing structured source bundles."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple, cast
from urllib.parse import urlsplit

import httpx

from .content_clusters import cluster_content
from .dedup import canonical_url, dedup_pages
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
from .queries import (
    focused_site_query,
    query_requires_decomposition,
    query_site_hosts,
    search_query_variants,
)
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
    _entity_lookup_queries,
    _future_outcome_is_unobservable,
    _has_historical_role_support,
    _insufficient_evidence_error,
    _query_anchor_terms,
    _sha256,
    _source_type,
)
from .scraper import WebPage, scrape_urls
from .search import (
    fetch_wikipedia_extract,
    search_web,
    search_wikipedia,
    wikipedia_lookup,
)
from .source_policy import matches_domain, normalize_domains, source_kind
from .temporal import period_match

_ACADEMIC_QUERY_RE = re.compile(
    r"\b(?:abstract|citation|doi|journal|paper|preprint|publication|study)\b",
    re.IGNORECASE,
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
    refinement_searches: int = 0,
    refinement_failures: int = 0,
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


async def gather_source_bundle(
    query: str,
    max_sources: int = 10,
    chars_per_source: int = 7000,
    client: Optional[httpx.AsyncClient] = None,
    ranker: RankingBackend = "lexical",
    render_thin_pages: bool = False,
    search_engine: str = "all",
    extractor: str = "bs4",
    jina_fallback: bool = False,
    include_academic: bool | None = None,
    preferred_domains: list[str] | None = None,
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
    site_hosts = query_site_hosts(clean_query)
    preferred_hosts = normalize_domains(preferred_domains or [])
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
    if not clean_query or len(clean_query) > 1000:
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
            error="query must not be empty."
            if not clean_query
            else "query must be at most 1000 characters.",
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
    refinement_searches = 0
    refinement_failures = 0
    candidates: List[WebPage] = []
    try:
        search_batches = []
        include_academic = (
            bool(_ACADEMIC_QUERY_RE.search(clean_query))
            if include_academic is None
            else include_academic
        )
        for search_query in search_queries:
            search_batches.append(
                await search_web(
                    search_query,
                    search_engine,
                    max_results=6,
                    client=client,
                    include_academic=include_academic,
                )
            )
        if preferred_hosts and not site_hosts:
            base_query = search_queries[-1]
            for domain in preferred_hosts[:2]:
                focused_query = f"{base_query} site:{domain}"
                search_queries.append(focused_query)
                refinement_searches += 1
                try:
                    search_batches.append(
                        await search_web(
                            focused_query,
                            search_engine,
                            max_results=4,
                            client=client,
                            include_academic=include_academic,
                        )
                    )
                except Exception:
                    refinement_failures += 1
        if len(site_hosts) > 1:
            result_hosts = {
                (urlsplit(result.url).hostname or "").lower()
                for batch in search_batches
                for result in batch
            }
            missing_hosts = [
                domain
                for domain in site_hosts
                if not any(
                    host == domain or host.endswith("." + domain)
                    for host in result_hosts
                )
            ]
            for domain in missing_hosts[:2]:
                focused_query = focused_site_query(clean_query, domain)
                search_queries.append(focused_query)
                refinement_searches += 1
                try:
                    search_batches.append(
                        await search_web(
                            focused_query,
                            search_engine,
                            max_results=6,
                            client=client,
                            include_academic=include_academic,
                        )
                    )
                except Exception:
                    refinement_failures += 1
        entity_batches = await asyncio.gather(
            *[
                search_wikipedia(entity_query, 2, client=client)
                for entity_query in _entity_lookup_queries(clean_query)
                if not site_hosts
                and entity_query.casefold()
                not in {query.casefold() for query in search_queries}
            ]
        )
        search_batches = [*entity_batches, *search_batches]
        seen_results = set()
        for batch in search_batches:
            for result in batch:
                if site_hosts:
                    host = (urlsplit(result.url).hostname or "").lower()
                    if not any(
                        domain and (host == domain or host.endswith("." + domain))
                        for domain in site_hosts
                    ):
                        continue
                result_key = canonical_url(result.url)
                if result_key not in seen_results:
                    seen_results.add(result_key)
                    results.append(result)
        if selected_ranker != "none":
            scores = lexical_relevance_scores(
                clean_query, [(result.title, result.snippet) for result in results]
            )
            results = [
                result
                for _, result in sorted(
                    zip(scores, results),
                    key=lambda item: (
                        -period_match(clean_query, item[1].title, item[1].url, None),
                        -int(
                            item[0] > 0
                            and matches_domain(
                                (urlsplit(item[1].url).hostname or "").lower(),
                                preferred_hosts,
                            )
                        ),
                        -item[0],
                    ),
                )
            ]
        if len(site_hosts) > 1:
            representatives = []
            for domain in site_hosts:
                match = next(
                    (
                        result
                        for result in results
                        if (urlsplit(result.url).hostname or "").lower() == domain
                        or (urlsplit(result.url).hostname or "")
                        .lower()
                        .endswith("." + domain)
                    ),
                    None,
                )
                if match and match not in representatives:
                    representatives.append(match)
            results = representatives + [
                result for result in results if result not in representatives
            ]
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
                extractor=extractor,
                jina_fallback=jina_fallback,
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
        good = [
            page
            for page in pages
            if page.text and len(page.text) > 150 and not page.error
        ]
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
            if (
                result.url not in scraped
                and result.snippet
                and len(result.snippet) > 120
            ):
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
        query_anchor_terms = _query_anchor_terms(clean_query)
        relevant_domains = {
            urlsplit(page.url).netloc.casefold().removeprefix("www.")
            for page in good
            if (
                len(page.text) > 200
                and page.content_origin != "search_snippet"
                and (
                    not query_anchor_terms
                    or query_anchor_terms <= lexical_query_terms(page.text)
                )
            )
            and urlsplit(page.url).netloc
        }
        if len(relevant_domains) == 1 and not multi_step_query and not site_hosts:
            excluded_domain = next(iter(relevant_domains))
            refinement_query = f"{search_queries[-1]} -site:{excluded_domain}"
            search_queries.append(refinement_query)
            refinement_searches += 1
            try:
                refinement_results = await search_web(
                    refinement_query,
                    search_engine,
                    max_results=6,
                    client=client,
                    include_academic=include_academic,
                )
            except Exception:
                refinement_results = []
                refinement_failures += 1
            new_results = []
            for result in refinement_results:
                result_key = canonical_url(result.url)
                if result_key in seen_results:
                    continue
                seen_results.add(result_key)
                results.append(result)
                new_results.append(result)
                if result.url.startswith("http"):
                    urls.append(result.url)
            refinement_urls = [
                result.url for result in new_results if result.url.startswith("http")
            ][:4]
            if refinement_urls:
                try:
                    refinement_pages = await scrape_urls(
                        refinement_urls,
                        max_chars=30000,
                        concurrency=4,
                        client=client,
                        js_render=render_thin_pages,
                        extractor=extractor,
                        jina_fallback=jina_fallback,
                    )
                except Exception:
                    refinement_pages = []
                    refinement_failures += 1
                attempted_urls.extend(refinement_urls)
                pages.extend(refinement_pages)
                retrieved_urls = {
                    page.url
                    for page in refinement_pages
                    if page.text and len(page.text) > 150 and not page.error
                }
                good.extend(
                    page for page in refinement_pages if page.url in retrieved_urls
                )
                for result in new_results:
                    if (
                        result.url in retrieved_urls
                        or not result.snippet
                        or len(result.snippet) <= 120
                    ):
                        continue
                    content_origin = (
                        "crossref_api"
                        if result.provider == "crossref"
                        else "search_snippet"
                    )
                    good.append(
                        WebPage(
                            result.url,
                            result.title,
                            result.snippet,
                            content_origin=content_origin,
                        )
                    )
                    if content_origin == "crossref_api":
                        metadata_fallbacks += 1
                    else:
                        snippet_fallbacks += 1
                good = dedup_pages(good)
        substantive = [page for page in good if len(page.text) > 200]
        if len(substantive) < 3 and not site_hosts:
            wiki_pages = await wikipedia_lookup(
                search_queries[-1], client=client, max_pages=2
            )
            wikipedia_fallbacks += len(wiki_pages)
            if wiki_pages:
                good = dedup_pages(good + wiki_pages)
                substantive = [page for page in good if len(page.text) > 200]
        candidates = good

    except Exception:
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
            refinement_searches=refinement_searches,
            refinement_failures=refinement_failures,
        )
        return SourceBundle(
            schema_version="1.6",
            bundle_id=_bundle_id(clean_query, "failed", []),
            query=clean_query,
            status="failed",
            sources=[],
            diagnostics=diagnostics,
            error="Retrieval failed. Check provider configuration or retry.",
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
    passage_size = min(2500, max(500, effective_chars_per_source // 3))
    max_passages_per_source = min(3, max(1, effective_chars_per_source // passage_size))

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
            len(page.text) > 200
            and page.content_origin != "search_snippet"
            and (not query_anchor_terms or query_anchor_terms <= source_terms)
        )
    ]
    substantive_sources = len(substantive_prepared)
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
        unique_domains=len(domains),
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
    bundle_status: BundleStatus = (
        "ok" if evidence_sufficiency == "sufficient" else "insufficient_evidence"
    )
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
    action = f"Recommended action: {bundle.diagnostics.recommended_action}."
    if bundle.status == "insufficient_evidence":
        return f"Evidence warning: {bundle.error}\n{action}\n\n{rendered}"
    return f"{action}\n\n{rendered}"
