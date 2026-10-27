"""Read-only projections of a research checkpoint for agent consumption."""

from __future__ import annotations

from typing import Any

from .research_errors import ResearchRequestError


def research_brief(snapshot: dict[str, Any]) -> dict[str, Any]:
    report = snapshot["report"]
    # Question gaps already appear in research_plan; do not repeat them here.
    plan_gaps = {
        f"{item['question_id']}: {gap}"
        for item in snapshot["research_plan"]["questions"]
        for gap in item["gaps"]
    }
    return {
        "schema_version": "1.0",
        "view": "brief",
        **{
            key: snapshot[key]
            for key in (
                "job_id",
                "revision",
                "question",
                "mode",
                "status",
                "remaining_steps",
                "elapsed_seconds",
                "source_policy",
                "preferred_domains",
                "research_plan",
                "claim_contract_version",
            )
        },
        "running_elsewhere": snapshot.get("running_elsewhere", False),
        "unsaved_result": snapshot.get("unsaved_result", False),
        "limitations": [
            note for note in snapshot["limitations"] if note not in plan_gaps
        ],
        "latest_attempt": snapshot["attempts"][-1] if snapshot["attempts"] else None,
        "sources": [
            {
                key: source[key]
                for key in (
                    "citation",
                    "source_id",
                    "url",
                    "title",
                    "passage_ids",
                    "content_origin",
                    "source_kind",
                    "retrieved_at",
                    "published_at",
                )
            }
            for source in snapshot["sources"]
        ],
        "source_catalog_count": len(snapshot["source_catalog"]),
        "report": {
            key: report[key]
            for key in (
                "status",
                "summary",
                "key_findings",
                "verification_scope",
                "limitations",
            )
        }
        if report
        else None,
        "evidence_access": {
            "tool": "get_research",
            "passages": {
                "job_id": snapshot["job_id"],
                "passage_ids": ["<passage ID>"],
            },
            "catalog": {
                "job_id": snapshot["job_id"],
                "view": "sources",
                "revision": snapshot["revision"],
                "offset": 0,
                "limit": 10,
            },
            "full": {"job_id": snapshot["job_id"], "view": "full"},
        },
    }


INLINE_PASSAGES = 12
INLINE_CHARS = 12000
INLINE_PASSAGE_CHARS = 1500


def _question_update(item: dict[str, Any]) -> dict[str, Any]:
    review = item["review"]
    return {
        **{key: value for key, value in item.items() if key != "review"},
        "review": {
            key: review[key]
            for key in ("status", "conclusion", "reviewed_revision")
            if key in review
        }
        | {"claims": len(review.get("claims", []))}
        if review
        else None,
    }


def _inline_sources(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """New sources with their passages inline, within a fixed text budget.

    Passages are taken round-robin across sources in their selected order, so
    each source shows its best passage before any shows a second. Longer
    passages are cut; read the full text by passage ID.
    """
    budget, count = INLINE_CHARS, 0
    inline: list[list[dict[str, Any]]] = [[] for _ in retrieved]
    for rank in range(max((len(s["passages"]) for s in retrieved), default=0)):
        for position, source in enumerate(retrieved):
            if rank >= len(source["passages"]) or count >= INLINE_PASSAGES:
                continue
            passage = source["passages"][rank]
            text = passage["text"][: min(INLINE_PASSAGE_CHARS, budget)]
            if not text:
                continue
            budget -= len(text)
            count += 1
            inline[position].append(
                {
                    "passage_id": passage["passage_id"],
                    "text": text,
                    "truncated": len(text) < len(passage["text"]),
                }
            )
    return [
        {
            **{key: value for key, value in source.items() if key != "passages"},
            "passages": passages,
            "omitted_passage_ids": [
                passage["passage_id"]
                for passage in source["passages"]
                if passage["passage_id"]
                not in {item["passage_id"] for item in passages}
            ],
        }
        for source, passages in zip(retrieved, inline)
    ]


def research_update(
    snapshot: dict[str, Any], question_id: str, retrieved: bool
) -> dict[str, Any]:
    """What changed for one question after a retrieval or an assessment.

    The full state stays in the brief (get_research); this carries the new
    sources with inline passage text, the affected question's gaps, claim
    support and counterevidence status, overall readiness and the next searches.
    """
    plan = snapshot["research_plan"]
    citations = {
        passage_id: source["citation"]
        for source in snapshot["sources"]
        for passage_id in source["passage_ids"]
    }
    attempt = snapshot["attempts"][-1] if snapshot["attempts"] else None
    update: dict[str, Any] = {
        "schema_version": "1.0",
        "view": "update",
        **{
            key: snapshot[key]
            for key in (
                "job_id",
                "revision",
                "status",
                "remaining_steps",
                "elapsed_seconds",
            )
        },
        "running_elsewhere": snapshot.get("running_elsewhere", False),
        "unsaved_result": snapshot.get("unsaved_result", False),
    }
    if retrieved:
        latest = snapshot.get("retrieved") or {"sources": [], "notes": []}
        update["attempt"] = attempt
        update["evidence_sufficiency"] = latest.get("evidence_sufficiency")
        update["retrieval_notes"] = latest["notes"]
        update["retrieved"] = [
            {
                **source,
                "citation": next(
                    (
                        citations[p["passage_id"]]
                        for p in source["passages"]
                        if p["passage_id"] in citations
                    ),
                    None,
                ),
            }
            for source in _inline_sources(latest["sources"])
        ]
    update.update(
        question=next(
            (
                _question_update(item)
                for item in plan["questions"]
                if item["question_id"] == question_id
            ),
            None,
        ),
        open_questions=[
            item["question_id"] for item in plan["questions"] if item["gaps"]
        ],
        ready_to_complete=plan["ready_to_complete"],
        next_action=plan["next_action"],
        suggested_queries=plan["suggested_queries"],
        evidence_access={
            "tool": "get_research",
            "passages": {"job_id": snapshot["job_id"], "passage_ids": ["<passage ID>"]},
            "brief": {"job_id": snapshot["job_id"], "view": "brief"},
        },
    )
    return update


def source_page(
    snapshot: dict[str, Any], offset: int, limit: int, revision: int | None
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 20:
        raise ResearchRequestError("Use offset >= 0 and a source page limit of 1-20.")
    if revision is not None and revision != snapshot["revision"]:
        raise ResearchRequestError(
            "Research revision changed. Reload the brief and restart source pagination.",
            code="stale_revision",
            field="revision",
        )
    catalog = snapshot["source_catalog"]
    end = min(offset + limit, len(catalog))
    return {
        "schema_version": "1.0",
        "view": "sources",
        "job_id": snapshot["job_id"],
        "revision": snapshot["revision"],
        "sources": catalog[offset:end],
        "total": len(catalog),
        "offset": offset,
        "next_offset": end if end < len(catalog) else None,
    }
