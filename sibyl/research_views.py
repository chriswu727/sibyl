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
