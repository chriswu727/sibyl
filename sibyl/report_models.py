"""Report contracts, independent of optional model and rendering providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Source:
    url: str
    title: str
    snippet: str
    relevance: str = ""
    supporting_snippet: str = ""  # exact evidence sentence for this citation
    source_id: str = ""
    passage_ids: List[str] = field(default_factory=list)
    retrieved_at: str = ""
    published_at: str | None = None
    content_origin: str = ""
    content_hash: str = ""
    source_kind: str = ""
    evidence: List[dict] = field(default_factory=list)


@dataclass
class FindingVerification:
    index: int
    supported: bool
    confidence: str = "low"
    cited: List[int] = field(default_factory=list)
    note: str = ""


@dataclass
class ResearchReport:
    query: str
    summary: str
    key_findings: List[str]
    sources: List[Source]
    analysis: str = ""
    predictions: str = ""
    cross_analysis: str = ""  # sentiment + consensus + disagreements
    confidence: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    search_queries: List[str] = field(default_factory=list)
    charts: List[str] = field(default_factory=list)
    market_data_summary: str = ""
    sub_questions: List[str] = field(default_factory=list)
    finding_verifications: List = field(
        default_factory=list
    )  # verifier.FindingVerification per finding
    status: str = "ok"  # ok | insufficient_evidence | failed
    error: str = ""
    schema_version: str = "1.0"
    limitations: List[str] = field(default_factory=list)
    verification_scope: str = "not_performed"
    research_stats: dict = field(default_factory=dict)
    research_plan: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ResearchReport:
        values = dict(data)
        values["timestamp"] = datetime.fromisoformat(values["timestamp"])
        values["sources"] = [Source(**source) for source in values["sources"]]
        values["finding_verifications"] = [
            FindingVerification(**verdict)
            for verdict in values.get("finding_verifications", [])
        ]
        return cls(**values)
