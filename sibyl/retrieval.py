"""Keyless retrieval pipeline producing structured source bundles."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

from .context import relevant_window_span
from .dedup import canonical_url, dedup_pages
from .evidence import (
    BundleDiagnostics,
    BundleStatus,
    EvidencePassage,
    EvidenceSource,
    SourceBundle,
)
from .passages import TextPassage, split_passages
from .ranking import lexical_relevance_scores
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
    ranking_method: str = "not_run",
    candidates_ranked: int = 0,
    chunks_ranked: int = 0,
    passages_returned: int = 0,
    passage_size: int = 0,
    max_passages_per_source: int = 0,
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
        ranking_method=ranking_method,
        candidates_ranked=candidates_ranked,
        chunks_ranked=chunks_ranked,
        passages_returned=passages_returned,
        passage_size=passage_size,
        max_passages_per_source=max_passages_per_source,
    )


async def gather_source_bundle(
    query: str,
    max_sources: int = 10,
    chars_per_source: int = 7000,
    client: Optional[httpx.AsyncClient] = None,
) -> SourceBundle:
    started_at = time.monotonic()
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
        )
        return SourceBundle(
            schema_version="1.2",
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
    if not clean_query:
        diagnostics = _diagnostics(
            requested_max_sources=requested_max_sources,
            effective_max_sources=effective_max_sources,
            requested_chars_per_source=requested_chars_per_source,
            effective_chars_per_source=effective_chars_per_source,
            started_at=started_at,
        )
        return SourceBundle(
            schema_version="1.2",
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
                good.append(WebPage(url=result.url, title=result.title, text=result.snippet))
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
        )
        return SourceBundle(
            schema_version="1.2",
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

    relevance_scores = lexical_relevance_scores(clean_query, ranking_documents)
    ranked_indices = sorted(
        range(len(source_candidates)),
        key=lambda index: (-relevance_scores[index], index),
    )
    selected_sources = [
        (*source_candidates[index], relevance_scores[index])
        for index in ranked_indices[:effective_max_sources]
    ]

    passage_size = min(2500, max(500, effective_chars_per_source // 3))
    max_passages_per_source = min(
        3, max(1, effective_chars_per_source // passage_size)
    )
    prepared = []
    chunks_ranked = 0
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
        chunk_scores = lexical_relevance_scores(
            clean_query, [(title, chunk.text) for chunk in chunks]
        )
        chunks_ranked += len(chunks)
        chunk_indices = sorted(
            range(len(chunks)), key=lambda index: (-chunk_scores[index], index)
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

    bundle_status: BundleStatus = "ok" if prepared else "insufficient_evidence"
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
        ranking_method="lexical_v1" if source_candidates else "not_run",
        candidates_ranked=len(source_candidates),
        chunks_ranked=chunks_ranked,
        passages_returned=sum(len(source.evidence) for source in sources),
        passage_size=passage_size if source_candidates else 0,
        max_passages_per_source=max_passages_per_source if source_candidates else 0,
    )
    return SourceBundle(
        schema_version="1.2",
        bundle_id=bundle_id,
        query=clean_query,
        status=bundle_status,
        sources=sources,
        diagnostics=diagnostics,
        error=(
            ""
            if sources
            else f"No sources found for query: {clean_query!r}. Try a different phrasing."
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
    return (
        f"Retrieved {len(bundle.sources)} sources for query {bundle.query!r}. "
        "Reason over these and "
        f"cite [Source N]; if the answer isn't here, gather more or say you don't know.\n\n"
        + "\n---\n".join(parts)
    )
