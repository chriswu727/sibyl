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


_BEFORE_ANSWERING = {
    "initial": "No readable source yet: search and read one",
    "primary_source": "Only background sources so far: read the original publisher",
    "independent_source": "All evidence is from one publisher: read an independent source",
    "counterevidence": (
        'No counterevidence read yet: read one with purpose="counterevidence"'
    ),
    "corroboration": "A fact rests on one source: read an independent one that states it",
}
ANSWER_GUIDANCE = (
    "Write the answer for the user from this material: cite the URLs, never Sibyl "
    "claim or passage IDs; state what the counterevidence showed and which gaps "
    "remain open."
)


def _before_answering(
    item: dict[str, Any], suggestions: list[dict[str, Any]]
) -> list[str]:
    """What to do for this question before answering, most important first."""
    steps = [
        f"{_BEFORE_ANSWERING[s['kind']]} (suggested search: {s['query']})."
        for s in suggestions
        if s["question_id"] == item["question_id"] and s["kind"] in _BEFORE_ANSWERING
    ]
    if item["counterevidence"]["status"] == "none_linked":
        steps.append(
            "Counterevidence was read but none is cited: say in the answer what it "
            "showed or why it does not apply."
        )
    if steps:
        steps.append(
            "Batch the URLs you find into one read: urls=[...], up to five, one "
            "retrieval step."
        )
    return steps


def _answer_material(
    item: dict[str, Any], catalog: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The question's recorded claims as user-facing lines with source URLs."""
    review = item["review"]
    if not review or not review.get("claims"):
        return None
    urls = {
        passage_id: source["url"]
        for source in catalog
        for passage_id in source["passage_ids"]
    }

    def cited(ids: list[str]) -> list[str]:
        return list(dict.fromkeys(urls[i] for i in ids if i in urls))

    challenging = review.get("challenging_passage_ids", [])
    return {
        "guidance": ANSWER_GUIDANCE,
        "claims": [
            {
                "text": claim["text"],
                "kind": claim["kind"],
                "status": claim["status"],
                "supporting_urls": cited(claim["supporting_passage_ids"]),
                "challenging_urls": cited(claim["challenging_passage_ids"]),
                **({"reason": claim["rationale"]} if claim.get("rationale") else {}),
            }
            for claim in review["claims"]
        ],
        "counterevidence": {
            "status": item["counterevidence"]["status"],
            "reads": item["counterevidence"]["reads"],
            "note": review.get("counterevidence_note", ""),
            "challenging_urls": cited(challenging),
        },
        "open_gaps": item["gaps"],
    }


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
    item = next(
        (entry for entry in plan["questions"] if entry["question_id"] == question_id),
        None,
    )
    update.update(
        question=_question_update(item) if item else None,
        before_answering=_before_answering(item, plan["suggested_queries"])
        if item
        else [],
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
    if not retrieved and item:
        update["answer_material"] = _answer_material(item, snapshot["source_catalog"])
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
