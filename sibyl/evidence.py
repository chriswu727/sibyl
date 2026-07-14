"""Structured evidence protocol shared by Sibyl retrieval consumers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional


BundleStatus = Literal["ok", "insufficient_evidence", "invalid_request", "failed"]


@dataclass(frozen=True)
class EvidencePassage:
    passage_id: str
    citation_id: str
    text: str
    content_hash: str
    score: Optional[float] = None


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    url: str
    title: str
    retrieved_at: str
    content_hash: str
    source_type: str
    char_count: int
    evidence: List[EvidencePassage]
    relevance_score: Optional[float] = None
    quality_score: Optional[float] = None


@dataclass(frozen=True)
class BundleDiagnostics:
    search_results: int
    unique_urls: int
    urls_attempted: int
    pages_scraped: int
    scrape_failures: int
    snippet_fallbacks: int
    wikipedia_fallbacks: int
    sources_returned: int
    requested_max_sources: int
    effective_max_sources: int
    requested_chars_per_source: int
    effective_chars_per_source: int
    latency_ms: int


@dataclass(frozen=True)
class SourceBundle:
    schema_version: Literal["1.0"]
    bundle_id: str
    query: str
    status: BundleStatus
    sources: List[EvidenceSource]
    diagnostics: BundleDiagnostics
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
