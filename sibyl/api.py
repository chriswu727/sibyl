"""Public keyless retrieval API."""
from __future__ import annotations

from typing import Optional

import httpx

from .evidence import SourceBundle
from .ranking import RankingBackend
from .retrieval import gather_source_bundle, render_source_bundle


async def gather_bundle(
    query: str,
    max_sources: int = 10,
    chars_per_source: int = 7000,
    *,
    ranker: RankingBackend = "lexical",
    client: Optional[httpx.AsyncClient] = None,
) -> SourceBundle:
    return await gather_source_bundle(
        query,
        max_sources=max_sources,
        chars_per_source=chars_per_source,
        ranker=ranker,
        client=client,
    )


async def gather_sources(
    query: str,
    max_sources: int = 10,
    chars_per_source: int = 7000,
    *,
    ranker: RankingBackend = "lexical",
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    bundle = await gather_bundle(
        query,
        max_sources=max_sources,
        chars_per_source=chars_per_source,
        ranker=ranker,
        client=client,
    )
    return render_source_bundle(bundle)
