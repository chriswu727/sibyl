"""Report contracts, independent of optional model and rendering providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Source:
    url: str
    title: str
    snippet: str
    relevance: str = ""
    supporting_snippet: str = ""  # exact evidence sentence for this citation


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
