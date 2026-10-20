"""Budgeted, provenance-preserving reads for an existing investigation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from .evidence import BundleDiagnostics, EvidencePassage, EvidenceSource, SourceBundle
from .passages import split_passages
from .scraper import scrape_url


async def read_source_bundle(url: str, query: str) -> SourceBundle:
    page = await scrape_url(url, max_chars=16000)
    text = page.text if not page.error else ""
    digest = hashlib.sha256(text.encode()).hexdigest()
    bundle_id = "read_" + hashlib.sha256((page.url + digest).encode()).hexdigest()[:20]
    usable = len(text.strip()) >= 200
    passages = [EvidencePassage(
        f"P{index}", f"{bundle_id}/S1/P{index}", chunk.text,
        hashlib.sha256(chunk.text.encode()).hexdigest(), chunk.start_char, chunk.end_char,
    ) for index, chunk in enumerate(split_passages(text, overlap_chars=0), 1)] if usable else []
    sources = [EvidenceSource(
        "S1", page.url, page.title, datetime.now(timezone.utc).isoformat(),
        digest, "web", len(text), passages, content_origin=page.content_origin,
        published_at=page.published_at, published_at_method=page.published_at_method,
    )] if usable else []
    return SourceBundle(
        "1.6", bundle_id, query, "ok" if usable else "insufficient_evidence", sources,
        BundleDiagnostics(
            0, 1, 1, int(usable), int(not usable), 0, 0, len(sources), 1, 1, 16000, 16000, 0,
            passages_returned=len(passages), evidence_chars=len(text),
            evidence_sufficiency="limited", recommended_action="refine_query",
            sufficiency_reasons=[
                "Direct source read; host must assess support, freshness and completeness."
                if usable else "No substantive text extracted; use an accessible HTML/full-text source."
            ],
        ),
        error="" if usable else "Source text unavailable or too thin; no evidence inferred.",
    )
