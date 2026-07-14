"""Structured evidence protocol shared by Sibyl retrieval consumers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


BundleStatus = Literal["ok", "insufficient_evidence", "invalid_request", "failed"]
EvidenceSufficiency = Literal["not_assessed", "sufficient", "limited", "insufficient"]


@dataclass(frozen=True)
class EvidencePassage:
    passage_id: str
    citation_id: str
    text: str
    content_hash: str
    start_char: int = 0
    end_char: int = 0
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
    requested_ranking_method: str = "lexical"
    ranking_method: str = "not_run"
    ranking_warning: str = ""
    candidates_ranked: int = 0
    chunks_ranked: int = 0
    passages_returned: int = 0
    passage_size: int = 0
    max_passages_per_source: int = 0
    coverage_method: str = "not_run"
    query_terms: int = 0
    matched_query_terms: int = 0
    query_term_coverage: Optional[float] = None
    unique_domains: int = 0
    substantive_sources: int = 0
    evidence_chars: int = 0
    evidence_sufficiency: EvidenceSufficiency = "not_assessed"
    sufficiency_reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceBundle:
    schema_version: Literal["1.5"]
    bundle_id: str
    query: str
    status: BundleStatus
    sources: List[EvidenceSource]
    diagnostics: BundleDiagnostics
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
