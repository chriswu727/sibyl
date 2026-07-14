"""Keyless retrieval pipeline producing structured source bundles."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, cast
from urllib.parse import urlsplit

import httpx

from .context import relevant_window_span
from .dedup import canonical_url, dedup_pages
from .evidence import (
    BundleDiagnostics,
    BundleStatus,
    EvidenceSufficiency,
    EvidencePassage,
    EvidenceSource,
    SourceBundle,
)
from .passages import TextPassage, split_passages
from .ranking import (
    RankingBackend,
    flashrank_relevance_scores,
    lexical_query_coverage,
    lexical_relevance_scores,
)
from .scraper import WebPage, scrape_urls
from .search import fetch_wikipedia_extract, search_web, wikipedia_lookup


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
    if blockers:
        return "insufficient", blockers

    limitations = []
    if substantive_sources < 2:
        limitations.append("fewer_than_two_substantive_sources")
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
    evidence_sufficiency: EvidenceSufficiency = "not_assessed",
    sufficiency_reasons: Optional[List[str]] = None,
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
        evidence_sufficiency=evidence_sufficiency,
        sufficiency_reasons=sufficiency_reasons or [],
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
    clean_query = str(query or "").strip()
    if requested_ranker not in {"lexical", "flashrank", "none"}:
        diagnostics = _diagnostics(
            requested_max_sources=requested_max_sources,
            effective_max_sources=effective_max_sources,
            requested_chars_per_source=requested_chars_per_source,
            effective_chars_per_source=effective_chars_per_source,
            started_at=started_at,
            requested_ranking_method=requested_ranker,
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
    candidates: List[WebPage] = []
    try:
        results = await search_web(
            clean_query, "all", max_results=6, client=client, include_academic=True
        )
        seen = set()
        for result in results:
            if result.url.startswith("http") and result.url not in seen:
                seen.add(result.url)
                urls.append(result.url)

        attempted_urls = urls[:max(effective_max_sources * 2, 12)]
        pages = await scrape_urls(
            attempted_urls,
            max_chars=30000,
            client=client,
            js_render=True,
        )
        good = [page for page in pages if page.text and len(page.text) > 150 and not page.error]
        scraped = {page.url for page in good}
        for result in results:
            if result.url not in scraped and result.snippet and len(result.snippet) > 120:
                good.append(
                    WebPage(
                        url=result.url,
                        title=result.title,
                        text=result.snippet,
                        content_origin="search_snippet",
                    )
                )
                snippet_fallbacks += 1

        good = dedup_pages(good)
        substantive = [page for page in good if len(page.text) > 200]
        if len(substantive) < 3:
            wiki_pages = await wikipedia_lookup(clean_query, client=client, max_pages=2)
            wikipedia_fallbacks = len(wiki_pages)
            if wiki_pages:
                good = dedup_pages(good + wiki_pages)
                substantive = [page for page in good if len(page.text) > 200]
        candidates = substantive if len(substantive) >= 3 else good

        wiki_indices = [
            index for index, page in enumerate(candidates)
            if "wikipedia.org/wiki/" in page.url
        ]
        if wiki_indices:
            extracts = await asyncio.gather(
                *[
                    fetch_wikipedia_extract(candidates[index].url, client)
                    for index in wiki_indices
                ],
                return_exceptions=True,
            )
            for index, extract in zip(wiki_indices, extracts):
                if isinstance(extract, str) and len(extract) > len(candidates[index].text):
                    candidates[index] = WebPage(
                        url=candidates[index].url,
                        title=candidates[index].title,
                        text=extract,
                        content_origin="wikipedia_api",
                        published_at=candidates[index].published_at,
                        published_at_method=candidates[index].published_at_method,
                    )
    except Exception as exc:
        diagnostics = _diagnostics(
            search_results=len(results),
            unique_urls=len(urls),
            urls_attempted=min(len(urls), max(effective_max_sources * 2, 12)),
            pages_scraped=len(pages),
            scrape_failures=sum(1 for page in pages if page.error),
            snippet_fallbacks=snippet_fallbacks,
            wikipedia_fallbacks=wikipedia_fallbacks,
            requested_max_sources=requested_max_sources,
            effective_max_sources=effective_max_sources,
            requested_chars_per_source=requested_chars_per_source,
            effective_chars_per_source=effective_chars_per_source,
            started_at=started_at,
            requested_ranking_method=requested_ranker,
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

    source_candidates = []
    ranking_documents = []
    for page in candidates:
        representative_start, representative_end = relevant_window_span(
            clean_query, page.text, width=effective_chars_per_source
        )
        representative = page.text[representative_start:representative_end]
        page_key = canonical_url(page.url)
        title = page.title or result_titles.get(page_key, page.url)
        source_hash = _sha256(page.text)
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
    selected_sources = [
        (*source_candidates[index], relevance_scores[index])
        for index in ranked_indices[:effective_max_sources]
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

    coverage = lexical_query_coverage(
        clean_query,
        [
            text
            for _, title, _, _, selected_passages in prepared
            for text in [title, *(chunk.text for chunk, _, _ in selected_passages)]
        ],
    )
    domains = {
        urlsplit(page.url).netloc.lower().removeprefix("www.")
        for page, _, _, _, _ in prepared
        if urlsplit(page.url).netloc
    }
    substantive_sources = sum(
        1 for page, _, _, _, _ in prepared if len(page.text) > 200
    )
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
    )
    bundle_status: BundleStatus = (
        "insufficient_evidence"
        if evidence_sufficiency == "insufficient"
        else "ok"
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
                relevance_score=relevance_score,
            )
        )

    attempted_count = min(len(urls), max(effective_max_sources * 2, 12))
    diagnostics = _diagnostics(
        search_results=len(results),
        unique_urls=len(urls),
        urls_attempted=attempted_count,
        pages_scraped=len(pages),
        scrape_failures=sum(1 for page in pages if page.error),
        snippet_fallbacks=snippet_fallbacks,
        wikipedia_fallbacks=wikipedia_fallbacks,
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
        evidence_sufficiency=evidence_sufficiency,
        sufficiency_reasons=sufficiency_reasons,
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
