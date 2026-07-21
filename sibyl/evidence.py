"""Structured evidence protocol shared by Sibyl retrieval consumers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


BundleStatus = Literal["ok", "insufficient_evidence", "invalid_request", "failed"]
EvidenceSufficiency = Literal["not_assessed", "sufficient", "limited", "insufficient"]
QueryComplexity = Literal["not_assessed", "single_step", "multi_step"]
RecommendedAction = Literal[
    "not_assessed",
    "synthesize",
    "refine_query",
    "decompose_query",
    "revise_request",
    "retry",
]
ContentOrigin = Literal[
    "direct_fetch",
    "jina_reader",
    "wikipedia_api",
    "search_snippet",
    "crossref_api",
]
PublishedAtMethod = Literal[
    "",
    "meta_article_published_time",
    "meta_date_published",
    "meta_citation_publication_date",
    "meta_dc_date_issued",
    "meta_date",
    "json_ld_date_published",
    "time_date_published",
    "jina_published_time",
]


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
    content_origin: ContentOrigin = "direct_fetch"
    published_at: Optional[str] = None
    published_at_method: PublishedAtMethod = ""
    content_cluster_id: str = ""
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
    candidate_content_clusters: int = 0
    duplicate_candidates: int = 0
    independent_content_clusters: int = 0
    duplicate_sources: int = 0
    content_cluster_method: str = "not_run"
    evidence_sufficiency: EvidenceSufficiency = "not_assessed"
    sufficiency_reasons: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    search_providers: List[str] = field(default_factory=list)
    max_source_query_term_coverage: Optional[float] = None
    metadata_fallbacks: int = 0
    query_complexity: QueryComplexity = "not_assessed"
    recommended_action: RecommendedAction = "not_assessed"


@dataclass(frozen=True)
class SourceBundle:
    schema_version: Literal["1.6"]
    bundle_id: str
    query: str
    status: BundleStatus
    sources: List[EvidenceSource]
    diagnostics: BundleDiagnostics
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
