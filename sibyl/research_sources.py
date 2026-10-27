"""Budgeted, provenance-preserving reads of host-supplied sources."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from .bundle_builder import Discovery, build_bundle
from .dedup import canonical_url
from .evidence import SourceBundle
from .queries import query_requires_decomposition
from .scraper import WebPage, scrape_url

MAX_READ_URLS = 5
READ_CHARS = 60000
EVIDENCE_CHARS_PER_SOURCE = 16000
PASSAGES_PER_SOURCE = 9


async def read_sources(
    urls: list[str],
    query: str,
    *,
    extractor: str = "bs4",
    jina_fallback: bool = False,
    js_render: bool = False,
) -> SourceBundle:
    """Read host-supplied URLs and assess them like discovered sources.

    Each page is read up to 60000 characters and contributes at most 16000
    evidence characters in up to nine query-selected passages. Passage
    selection, content clustering, ranking and the sufficiency assessment are
    the same as for Sibyl's own search results.
    """
    started_at = time.monotonic()
    unique = list(dict.fromkeys(urls))
    if not unique or len(unique) > MAX_READ_URLS:
        raise ValueError(f"Read between 1 and {MAX_READ_URLS} source URLs at a time.")
    pages: list[WebPage] = list(
        await asyncio.gather(
            *(
                scrape_url(
                    url,
                    max_chars=READ_CHARS,
                    extractor=extractor,
                    jina_fallback=jina_fallback,
                    js_render=js_render,
                )
                for url in unique
            )
        )
    )
    candidates: list[WebPage] = []
    seen: set[str] = set()
    for page in pages:
        key = canonical_url(page.url)
        if page.error or len(page.text.strip()) < 200 or key in seen:
            continue
        seen.add(key)
        candidates.append(page)
    bundle = await build_bundle(
        query.strip(),
        candidates,
        Discovery(urls=unique, attempted_urls=unique, pages=pages),
        requested_max_sources=len(unique),
        effective_max_sources=len(unique),
        requested_chars_per_source=EVIDENCE_CHARS_PER_SOURCE,
        effective_chars_per_source=EVIDENCE_CHARS_PER_SOURCE,
        requested_ranker="lexical",
        selected_ranker="lexical",
        preferred_hosts=[],
        multi_step_query=query_requires_decomposition(query),
        started_at=started_at,
        max_passages_per_source=PASSAGES_PER_SOURCE,
        status_from_read=True,
    )
    failures = [
        page.error
        or "No substantive text extracted; use an accessible HTML/full-text source."
        for page in pages
        if page.error or len(page.text.strip()) < 200
    ]
    notes = (
        ["Direct source read; host must assess support, freshness and completeness."]
        if bundle.sources
        else []
    )
    notes += list(dict.fromkeys(failures))
    if any(len(page.text) > EVIDENCE_CHARS_PER_SOURCE for page in candidates):
        notes.append(
            "Query-selected excerpts from a bounded read of up to 60000 characters; omitted sections remain unread."
        )
    diagnostics = replace(
        bundle.diagnostics,
        sufficiency_reasons=list(
            dict.fromkeys([*notes, *bundle.diagnostics.sufficiency_reasons])
        ),
    )
    error = (
        ""
        if bundle.sources
        else (
            failures[0]
            if failures
            else "Source text unavailable or too thin; no evidence inferred."
        )
    )
    return replace(bundle, diagnostics=diagnostics, error=error)


async def read_source_bundle(url: str, query: str) -> SourceBundle:
    """Read one host-supplied URL; see read_sources."""
    return await read_sources([url], query)
