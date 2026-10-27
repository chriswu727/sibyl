"""Persistent research records and host review contracts."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from .evidence import BundleStatus

JobStatus = Literal[
    "awaiting_input",
    "running",
    "interrupted",
    "completed",
    "cancelled",
    "failed",
    "budget_exhausted",
]
ResearchMode = Literal["host", "autonomous"]
ReviewStatus = Literal["supported", "disputed", "unresolved"]
SearchPurpose = Literal["explore", "counterevidence", "follow_citation"]


class RetrievalAttempt(TypedDict):
    query: str
    status: BundleStatus | Literal["running"]
    round: int
    bundle_id: NotRequired[str]
    substantive_sources: NotRequired[int]
    error: NotRequired[str]
    question_id: NotRequired[str]
    purpose: NotRequired[SearchPurpose]
    url: NotRequired[str]
    urls: NotRequired[list[str]]
    parent_passage_id: NotRequired[str]
    rationale: NotRequired[str]
    based_on_revision: NotRequired[int]
    prior_gaps: NotRequired[list[str]]
    prior_review_status: NotRequired[ReviewStatus | None]


class EvidenceReview(TypedDict):
    question_id: str
    status: ReviewStatus
    conclusion: str
    supporting_passage_ids: list[str]
    challenging_passage_ids: list[str]
    counterevidence_note: str
    reviewed_attempts: int
    assessment_origin: Literal["host"]
    follow_up_queries: list[str]
    claims: NotRequired[list[dict[str, Any]]]
    basis_fingerprints: NotRequired[dict[str, str]]


class QuestionPlan(TypedDict):
    question_id: str
    question: str
    review: EvidenceReview | None
    gaps: list[str]
    attempts: int


class SuggestedQuery(TypedDict):
    question_id: str
    purpose: SearchPurpose
    query: str
    reason: str


class ResearchPlan(TypedDict):
    questions: list[QuestionPlan]
    ready_to_complete: bool
    next_action: str
    remaining_steps: int
    assessment_scope: str
    suggested_queries: list[SuggestedQuery]


class JobEvent(TypedDict):
    message: str
    time: float


class JobPayload(TypedDict):
    question: str
    mode: ResearchMode
    depth: int
    language: str
    status: JobStatus
    notebook: dict[str, Any]
    report: dict[str, Any] | None
    events: list[JobEvent]
    elapsed_seconds: float


class JobOwner(TypedDict):
    instance_id: str
    claimed_at: float


class JobRecord(JobPayload):
    job_id: str
    revision: int
    # Present only while running: the runtime instance that claimed the job.
    owner: NotRequired[JobOwner]
    # Read-time projections for views; never persisted.
    running_elsewhere: NotRequired[bool]
    unsaved_result: NotRequired[bool]
