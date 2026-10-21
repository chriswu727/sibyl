"""Host-declared evidence reviews with explicit structural readiness gates."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .evidence import EvidencePassage, EvidenceSource
from .job_models import (
    EvidenceReview,
    QuestionPlan,
    ResearchPlan,
    RetrievalAttempt,
    ReviewStatus,
    SearchPurpose,
    SuggestedQuery,
)

if TYPE_CHECKING:
    from .research_state import ResearchNotebook

from .research_errors import ResearchRequestError
from .source_policy import source_kind


def questions(notebook: ResearchNotebook) -> dict[str, str]:
    return {
        f"Q{index}": text for index, text in enumerate(notebook.sub_questions, 1)
    } or {"Q0": notebook.question}


def evidence_index(
    notebook: ResearchNotebook,
) -> dict[str, tuple[EvidenceSource, EvidencePassage]]:
    return {
        passage.citation_id: (source, passage)
        for bundle in notebook.bundles
        for source in bundle.sources
        if source.content_origin != "search_snippet"
        for passage in source.evidence
    }


def question_attempts(
    notebook: ResearchNotebook, question_id: str, question: str
) -> list[RetrievalAttempt]:
    return [
        a
        for a in notebook.attempts
        if a.get("question_id") == question_id
        or (not a.get("question_id") and a["query"] == question)
    ]


def review_question(
    notebook: ResearchNotebook,
    question_id: str,
    status: str,
    supporting: list[str],
    challenging: list[str],
    conclusion: str,
    counterevidence_note: str,
    follow_up_queries: list[str] | None = None,
) -> None:
    if question_id not in questions(notebook):
        raise ResearchRequestError(
            "Unknown research question ID; read the research plan first."
        )
    if status not in {"supported", "disputed", "unresolved"}:
        raise ResearchRequestError(
            "Review status must be supported, disputed, or unresolved."
        )
    if not conclusion.strip() or not counterevidence_note.strip():
        raise ResearchRequestError(
            "Explain the conclusion and the counterevidence investigation, including gaps."
        )
    if (
        max(len(conclusion), len(counterevidence_note)) > 3000
        or len(supporting) + len(challenging) > 24
    ):
        raise ResearchRequestError(
            "Review exceeds the bounded evidence or prose limit."
        )
    if set(supporting) & set(challenging):
        raise ResearchRequestError(
            "The same passage cannot both support and challenge this review."
        )
    index = evidence_index(notebook)
    if any(item not in index for item in [*supporting, *challenging]):
        raise ResearchRequestError(
            "Review references an unknown or snippet-only passage."
        )
    if status == "supported" and (not supporting or challenging):
        raise ResearchRequestError(
            "Supported reviews require support and no unresolved challenging passages."
        )
    if status == "disputed" and (not supporting or not challenging):
        raise ResearchRequestError(
            "Disputed reviews require both supporting and challenging evidence."
        )
    follow_ups = follow_up_queries or []
    if len(follow_ups) > 4 or any(
        not query.strip() or len(query) > 1000 for query in follow_ups
    ):
        raise ResearchRequestError(
            "Use at most four nonempty follow-up queries of at most 1000 characters."
        )
    review: EvidenceReview = dict(
        question_id=question_id,
        status=cast(ReviewStatus, status),
        conclusion=conclusion.strip(),
        supporting_passage_ids=list(dict.fromkeys(supporting)),
        challenging_passage_ids=list(dict.fromkeys(challenging)),
        counterevidence_note=counterevidence_note.strip(),
        reviewed_attempts=len(notebook.attempts),
        assessment_origin="host",
        follow_up_queries=list(dict.fromkeys(query.strip() for query in follow_ups)),
    )
    updated = {**notebook.reviews, question_id: review}
    urls = {
        index[item][0].url
        for item_review in updated.values()
        for item in [
            *item_review["supporting_passage_ids"],
            *item_review["challenging_passage_ids"],
        ]
    }
    if len(urls) > notebook.synthesis_limit:
        raise ResearchRequestError(
            "Reviewed sources exceed the report window; narrow the evidence links."
        )
    previous = notebook.reviews
    notebook.reviews = updated
    selected_refs = {
        p.citation_id for source in notebook.selected_sources() for p in source.evidence
    }
    all_refs = {
        ref
        for item_review in updated.values()
        for ref in [
            *item_review["supporting_passage_ids"],
            *item_review["challenging_passage_ids"],
        ]
    }
    if not all_refs <= selected_refs:
        notebook.reviews = previous
        raise ResearchRequestError(
            "Review links duplicate or conflicting source snapshots; select one snapshot per source/content cluster."
        )


def _suggested_queries(
    notebook: ResearchNotebook, items: list[QuestionPlan]
) -> list[SuggestedQuery]:
    suggestions: list[SuggestedQuery] = []
    for item in items:
        if not item["gaps"]:
            continue
        attempts = question_attempts(notebook, item["question_id"], item["question"])
        completed = {
            " ".join(a["query"].casefold().split())
            for a in attempts
            if a["status"] in {"ok", "insufficient_evidence"} and not a.get("url")
        }
        review = item["review"]
        follow_ups = review["follow_up_queries"] if review else []
        candidates: list[tuple[str, SearchPurpose, str]] = [
            (query, "explore", "host_review_gap") for query in follow_ups
        ]
        if not follow_ups:
            suffix = (
                " original research paper methods results limitations"
                if notebook.source_policy == "academic"
                else " primary source official documentation"
            )
            candidates.append(
                (item["question"] + suffix, "explore", "research_plan_gap")
            )
        candidates.append(
            (
                item["question"] + " limitations conflicting evidence replication",
                "counterevidence",
                "counterevidence_check",
            )
        )
        for query, purpose, reason in candidates:
            normalized = " ".join(query.casefold().split())
            if normalized not in completed:
                suggestions.append(
                    dict(
                        question_id=item["question_id"],
                        purpose=purpose,
                        query=query,
                        reason=reason,
                    )
                )
                completed.add(normalized)
    return suggestions


def research_plan(notebook: ResearchNotebook) -> ResearchPlan:
    index = evidence_index(notebook)
    items: list[QuestionPlan] = []
    for question_id, question in questions(notebook).items():
        attempts = question_attempts(notebook, question_id, question)
        review = notebook.reviews.get(question_id)
        gaps = []
        if not attempts:
            gaps.append("No investigation linked to this question.")
        if not review:
            gaps.append("Evidence has not been assessed by the host.")
        else:
            if review["status"] == "unresolved":
                gaps.append("The host marked this question unresolved.")
            if any(
                i >= review["reviewed_attempts"] and a in attempts
                for i, a in enumerate(notebook.attempts)
            ):
                gaps.append(
                    "New retrieval after the review; reassess before completing."
                )
            refs = [
                *review["supporting_passage_ids"],
                *review["challenging_passage_ids"],
            ]
            substantive = [index[ref][0] for ref in refs if ref in index]
            if not substantive or all(
                source_kind(s, notebook.preferred_domains)
                in {"encyclopedia", "community", "scholarly_metadata"}
                for s in substantive
            ):
                gaps.append(
                    "Substantive evidence beyond background/metadata is missing; inspect primary full text."
                )
        if not any(
            a.get("purpose") == "counterevidence"
            and a.get("substantive_sources", 0) > 0
            and a["status"] in {"ok", "insufficient_evidence"}
            for a in attempts
        ):
            gaps.append(
                "No completed counterevidence search/read is linked to this question."
            )
        items.append(
            dict(
                question_id=question_id,
                question=question,
                review=review,
                gaps=gaps,
                attempts=len(attempts),
            )
        )
    ready = all(not item["gaps"] for item in items)
    remaining = max(0, notebook.max_steps - len(notebook.attempts))
    return dict(
        questions=items,
        ready_to_complete=ready,
        next_action="synthesize"
        if ready
        else ("review_or_report_limitations" if not remaining else "investigate_gaps"),
        remaining_steps=remaining,
        assessment_scope="host_declared_evidence_links_not_semantic_verification",
        suggested_queries=_suggested_queries(notebook, items) if remaining else [],
    )
