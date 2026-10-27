"""Keyless retrieval pipeline producing structured source bundles."""

from __future__ import annotations

import asyncio
import re
import time
from typing import List, Optional, cast
from urllib.parse import urlsplit

import httpx

from .bundle_builder import Discovery, _diagnostics, _has_substantive_text, build_bundle
from .dedup import canonical_url, dedup_pages
from .evidence import (
    QueryComplexity,
    SourceBundle,
)
from .queries import (
    focused_site_query,
    query_requires_decomposition,
    query_site_hosts,
    search_query_variants,
)
from .ranking import (
    RankingBackend,
    lexical_query_terms,
    lexical_relevance_scores,
)
from .retrieval_policy import (
    _bundle_id,
    _entity_lookup_queries,
    _query_anchor_terms,
)
from .scraper import WebPage, scrape_urls
from .search import (
    fetch_wikipedia_extract,
    search_web,
    search_wikipedia,
    wikipedia_lookup,
)
from .source_policy import matches_domain, normalize_domains
from .temporal import period_match

_ACADEMIC_QUERY_RE = re.compile(
    r"\b(?:abstract|citation|doi|journal|paper|preprint|publication|study)\b",
    re.IGNORECASE,
)


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
                _has_substantive_text(page)
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
        substantive = [page for page in good if _has_substantive_text(page)]
        if len(substantive) < 3 and not site_hosts:
            wiki_pages = await wikipedia_lookup(
                search_queries[-1], client=client, max_pages=2
            )
            wikipedia_fallbacks += len(wiki_pages)
            if wiki_pages:
                good = dedup_pages(good + wiki_pages)
                substantive = [page for page in good if _has_substantive_text(page)]
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

    return await build_bundle(
        clean_query,
        candidates,
        Discovery(
            results=results,
            urls=urls,
            attempted_urls=attempted_urls,
            pages=pages,
            snippet_fallbacks=snippet_fallbacks,
            wikipedia_fallbacks=wikipedia_fallbacks,
            metadata_fallbacks=metadata_fallbacks,
            search_queries=search_queries,
            refinement_searches=refinement_searches,
            refinement_failures=refinement_failures,
        ),
        requested_max_sources=requested_max_sources,
        effective_max_sources=effective_max_sources,
        requested_chars_per_source=requested_chars_per_source,
        effective_chars_per_source=effective_chars_per_source,
        requested_ranker=requested_ranker,
        selected_ranker=selected_ranker,
        preferred_hosts=preferred_hosts,
        multi_step_query=multi_step_query,
        started_at=started_at,
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
