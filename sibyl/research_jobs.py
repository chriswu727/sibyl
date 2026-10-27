"""Durable host-led research orchestration."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from .job_models import JobRecord, ResearchMode, SearchPurpose

if TYPE_CHECKING:
    from .runtime import SibylRuntime

from .async_cache import CapacityError
from .job_store import JobStore
from .privacy import display_url
from .report_audit import citation_errors
from .report_models import ResearchReport
from .research_claims import render_claims
from .research_errors import ResearchRequestError
from .research_review import evidence_index, questions, research_plan, review_question
from .research_sources import read_source_bundle
from .research_state import ResearchBudgetExceeded, ResearchNotebook
from .retrieval import gather_source_bundle
from .source_policy import normalize_domains, source_kind


class ResearchJobs:
    def __init__(self, runtime: SibylRuntime) -> None:
        self.runtime = runtime
        self.store = JobStore(runtime.policy.output_root)
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.closed = False

    def _event(self, job: JobRecord, message: str) -> None:
        job["events"] = [
            *job["events"][-79:],
            {"message": message[:240], "time": time.time()},
        ]

    def _checkpoint(self, job: JobRecord) -> None:
        try:
            self.store.save(job)
        except CapacityError:
            retained = self.store.get(job["job_id"])
            previous = retained["notebook"]
            current = job["notebook"]
            previous["model_calls"] = current["model_calls"]
            previous["reserved_output_tokens"] = current["reserved_output_tokens"]
            previous["attempts"] = current["attempts"]
            retained_ids = {bundle["bundle_id"] for bundle in previous["bundles"]}
            for attempt in previous["attempts"]:
                if (
                    attempt["status"] == "running"
                    or attempt.get("bundle_id", "") not in retained_ids
                ):
                    attempt.update(status="failed", bundle_id="", substantive_sources=0)
                    attempt.pop("error", None)
            retained["elapsed_seconds"] = job["elapsed_seconds"]
            retained["status"] = "awaiting_input" if job["mode"] == "host" else "failed"
            if job["status"] in {"interrupted", "budget_exhausted"}:
                retained["status"] = job["status"]
            self._event(
                retained,
                "Checkpoint size limit reached; new evidence/report not saved. Prior evidence and consumed budgets retained.",
            )
            self.store.save(retained, recovery=True)
            cast(dict[str, Any], job).clear()
            job.update(retained)
            raise CapacityError(
                "Checkpoint size limit reached; prior evidence and consumed budgets retained."
            ) from None

    def _idle(self, job: JobRecord) -> None:
        if self.closed:
            raise ValueError("Research runtime is closed.")
        if job["job_id"] in self.tasks:
            raise CapacityError("Research job is busy.")
        if job["status"] == "completed":
            raise ResearchRequestError(
                "Completed reports are immutable. Start a new research job.",
                code="immutable_report",
            )

    def start(
        self,
        question: str,
        sub_questions: list[str],
        mode: str,
        depth: int,
        language: str,
        max_steps: int,
        source_policy: str = "primary",
        preferred_domains: list[str] | None = None,
        claim_contract_version: int = 1,
    ) -> dict[str, Any]:
        self.runtime.validate_query(question)
        if mode != "host":
            raise ResearchRequestError(
                "Sibyl no longer calls models. Use host mode from your MCP agent."
            )
        if not question.strip() or depth not in {1, 2, 3}:
            raise ValueError("Invalid research request.")
        if not 1 <= max_steps <= 12 or len(sub_questions) > 5:
            raise ValueError("Use 1-12 retrieval steps and at most five sub-questions.")
        if source_policy not in {"primary", "academic"}:
            raise ValueError("Source policy must be primary or academic.")
        if claim_contract_version not in {1, 2}:
            raise ResearchRequestError("Unknown research result contract version.")
        preferred_domains = normalize_domains(preferred_domains or [])
        for item in sub_questions:
            self.runtime.validate_query(item)
            if not item.strip():
                raise ValueError("Sub-questions must not be empty.")
        notebook = ResearchNotebook(
            question,
            sub_questions=list(dict.fromkeys(sub_questions)),
            max_steps=max_steps,
            source_policy=source_policy,
            preferred_domains=preferred_domains,
            require_review=True,
            claim_contract_version=claim_contract_version,
        )
        job = self.store.create(
            {
                "question": question,
                "mode": cast(ResearchMode, mode),
                "depth": depth,
                "language": language,
                "status": "awaiting_input",
                "notebook": notebook.to_dict(),
                "report": None,
                "events": [],
                "elapsed_seconds": 0.0,
            }
        )
        self._event(job, "Research created; sources are untrusted data.")
        self.store.save(job)
        return self.get(job["job_id"])

    def _load(self, job_id: str) -> tuple[JobRecord, ResearchNotebook]:
        job = self.store.get(job_id)
        if job["status"] == "running" and job_id not in self.tasks:
            job["status"] = "interrupted"
        return job, ResearchNotebook.from_dict(job["notebook"])

    def _passage(
        self, job: JobRecord, notebook: ResearchNotebook, passage_id: str
    ) -> dict[str, Any]:
        item = evidence_index(notebook).get(passage_id)
        if item is None:
            raise ValueError("Unknown evidence passage.")
        source, passage = item
        job_id = job["job_id"]
        return dict(
            job_id=job_id,
            revision=job["revision"],
            url=display_url(source.url),
            title=source.title,
            passage=passage.__dict__,
            source_char_count=source.char_count,
            is_excerpt=passage.start_char > 0 or passage.end_char < source.char_count,
            read_more={
                "tool": "continue_research",
                "job_id": job_id,
                "url": display_url(source.url),
                "requires_focused_query": True,
                "uses_retrieval_step": True,
            }
            if job["mode"] == "host"
            and job["status"] in {"awaiting_input", "interrupted"}
            and len(notebook.attempts) < notebook.max_steps
            else None,
            source_kind=source_kind(source, notebook.preferred_domains),
            source_id=source.source_id,
            content_hash=source.content_hash,
            content_origin=source.content_origin,
            content_cluster_id=source.content_cluster_id,
            retrieved_at=source.retrieved_at,
            published_at=source.published_at,
        )

    def passages(self, job_id: str, passage_ids: list[str]) -> dict[str, Any]:
        """Read several exact passages in one call, in the requested order."""
        passage_ids = list(dict.fromkeys(passage_ids))
        if not 1 <= len(passage_ids) <= 12:
            raise ResearchRequestError(
                "Read 1-12 passage IDs at a time.", field="passage_ids"
            )
        job, notebook = self._load(job_id)
        items = [self._passage(job, notebook, key) for key in passage_ids]
        for item in items:
            del item["job_id"], item["revision"]
        return {"job_id": job_id, "revision": job["revision"], "passages": items}

    def get(self, job_id: str, passage_id: str = "") -> dict[str, Any]:
        job, notebook = self._load(job_id)
        if passage_id:
            return self._passage(job, notebook, passage_id)
        public: dict[str, Any] = {
            key: value for key, value in job.items() if key != "notebook"
        }
        public.update(
            sub_questions=notebook.sub_questions,
            attempts=notebook.attempts,
            remaining_steps=max(0, notebook.max_steps - len(notebook.attempts)),
            model_calls=notebook.model_calls,
            reserved_output_tokens=notebook.reserved_output_tokens,
            sources=[
                {"citation": f"[Source {index}]", **source}
                for index, source in enumerate(
                    [source.__dict__ for source in notebook.report_sources()], 1
                )
            ],
            evidence=[
                source.__dict__ | {"evidence": [p.__dict__ for p in source.evidence]}
                for source in notebook.selected_sources()
            ],
            limitations=notebook.limitations(),
            research_plan=research_plan(notebook),
            source_policy=notebook.source_policy,
            preferred_domains=notebook.preferred_domains,
            claim_contract_version=notebook.claim_contract_version,
            source_catalog=[
                dict(
                    source_id=source.source_id,
                    content_hash=source.content_hash,
                    content_cluster_id=source.content_cluster_id,
                    url=display_url(source.url),
                    title=source.title,
                    source_kind=source_kind(source, notebook.preferred_domains),
                    content_origin=source.content_origin,
                    retrieved_at=source.retrieved_at,
                    published_at=source.published_at,
                    passage_ids=[p.citation_id for p in source.evidence],
                )
                for bundle in notebook.bundles
                for source in bundle.sources
            ],
        )
        for source in [*public["sources"], *public["evidence"]]:
            source["url"] = display_url(source["url"])
        if public["report"]:
            for source in public["report"]["sources"]:
                source["url"] = display_url(source["url"])
        return public

    async def continue_job(
        self,
        job_id: str,
        query: str,
        url: str = "",
        question_id: str = "",
        purpose: str = "explore",
        parent_passage_id: str = "",
        rationale: str = "",
    ) -> dict[str, Any]:
        job = self.store.get(job_id)
        self._idle(job)
        if job["mode"] != "host":
            raise ResearchRequestError(
                "Autonomous execution has been removed. Read or export the saved job; start a new host research job to continue investigating."
            )
        self.runtime.validate_query(query)
        if not query.strip() and not url:
            raise ValueError(
                "Host-assisted research requires a focused query or source URL."
            )
        if purpose not in {"explore", "counterevidence", "follow_citation"}:
            raise ValueError("Unknown investigation purpose.")
        if len(url) > 2000:
            raise ValueError("Source URL is too long.")
        if len(rationale) > 1000:
            raise ResearchRequestError(
                "Retrieval rationale must be at most 1000 characters."
            )
        if len(self.tasks) >= 2:
            raise CapacityError("Research worker capacity reached.")
        remaining = self.runtime.policy.report_timeout - job["elapsed_seconds"]
        if remaining <= 0:
            raise ResearchBudgetExceeded("Research wall-time budget exhausted.")
        notebook = ResearchNotebook.from_dict(job["notebook"])
        plan_questions = questions(notebook)
        question_id = question_id or next(
            (key for key, value in plan_questions.items() if value == query), ""
        )
        if not question_id and len(plan_questions) == 1:
            question_id = next(iter(plan_questions))
        if question_id not in plan_questions:
            raise ValueError("Choose a question_id from the research plan.")
        if parent_passage_id and parent_passage_id not in evidence_index(notebook):
            raise ValueError("Parent passage must belong to this investigation.")
        if purpose == "follow_citation" and (not url or not parent_passage_id):
            raise ValueError("Citation following requires a URL and parent passage ID.")
        query = query.strip() or plan_questions[question_id]
        previous_question = next(
            item
            for item in research_plan(notebook)["questions"]
            if item["question_id"] == question_id
        )
        notebook.begin_search(query)
        notebook.attempts[-1].update(
            {
                "question_id": question_id,
                "purpose": cast(SearchPurpose, purpose),
                "url": display_url(url) if url else "",
                "parent_passage_id": parent_passage_id,
                "rationale": rationale.strip(),
                "based_on_revision": job["revision"],
                "prior_gaps": previous_question["gaps"],
                "prior_review_status": previous_question["review"]["status"]
                if previous_question["review"]
                else None,
            }
        )
        job["notebook"] = notebook.to_dict()
        job["status"] = "running"
        self._event(job, f"Retrieving evidence: {query}")
        self.store.save(job)
        started = time.monotonic()
        task = asyncio.current_task()
        assert task is not None
        self.tasks[job_id] = task
        try:
            bundle = await asyncio.wait_for(
                read_source_bundle(url, query)
                if url
                else gather_source_bundle(
                    query,
                    max_sources=8,
                    chars_per_source=4000,
                    include_academic=True
                    if notebook.source_policy == "academic"
                    else None,
                    preferred_domains=notebook.preferred_domains,
                ),
                min(remaining, self.runtime.policy.tool_timeout * 0.9),
            )
            notebook.finish_search(query, bundle)
            job["status"] = "awaiting_input"
            self._event(
                job,
                f"Retrieval finished: {bundle.status}; inspect evidence before writing.",
            )
        except asyncio.CancelledError:
            notebook.finish_search(query, None)
            job["status"] = "interrupted"
            raise
        except Exception:
            notebook.finish_search(query, None)
            job["status"] = "awaiting_input"
            self._event(
                job,
                "Retrieval failed; no evidence inferred. Retry or refine explicitly.",
            )
        finally:
            job["notebook"] = notebook.to_dict()
            job["elapsed_seconds"] += time.monotonic() - started
            try:
                self._checkpoint(job)
            finally:
                self.tasks.pop(job_id, None)
        return self.get(job_id)

    def assess(
        self,
        job_id: str,
        revision: int,
        question_id: str,
        status: str,
        supporting: list[str],
        challenging: list[str],
        conclusion: str,
        counterevidence_note: str,
        follow_up_queries: list[str] | None = None,
        claims: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        job = self.store.get(job_id)
        self._idle(job)
        if job["mode"] != "host":
            raise ResearchRequestError(
                "Legacy autonomous jobs cannot be assessed. Read/export the saved state or start a new host job.",
                code="unsupported_mode",
                field="mode",
            )
        if revision != job["revision"]:
            raise ResearchRequestError(
                "Reload the host research job and use its current revision.",
                code="stale_revision",
                field="revision",
            )
        notebook = ResearchNotebook.from_dict(job["notebook"])
        review_question(
            notebook,
            question_id,
            status,
            supporting,
            challenging,
            conclusion,
            counterevidence_note,
            follow_up_queries,
            claims,
        )
        job["notebook"] = notebook.to_dict()
        self._event(
            job,
            f"Host reviewed {question_id}: {status}; semantic support is not independently verified.",
        )
        self.store.save(job)
        return self.get(job_id)

    def complete(
        self,
        job_id: str,
        revision: int,
        summary: str,
        key_findings: list[str],
        analysis: str,
        limitations: list[str],
        allow_incomplete: bool = False,
        claim_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        import re

        job = self.store.get(job_id)
        self._idle(job)
        if job["mode"] != "host":
            raise ResearchRequestError(
                "Legacy autonomous jobs cannot be finalized. Read/export the saved state or start a new host job.",
                code="unsupported_mode",
                field="mode",
            )
        if job["revision"] != revision:
            raise ResearchRequestError(
                "Reload the host research job and use its current revision.",
                code="stale_revision",
                field="revision",
            )
        if len(key_findings) > 32 or len(limitations) > 32:
            raise ValueError("At most 32 findings and limitations are supported.")
        notebook = ResearchNotebook.from_dict(job["notebook"])
        report_claims: list[dict[str, Any]] = []
        if notebook.claim_contract_version >= 2:
            if summary or key_findings or analysis:
                raise ResearchRequestError(
                    "This job uses evidence-linked claims. Record prose with assess_research(claims=...), then complete with claim_ids and omit summary/key_findings/analysis."
                )
            report_claims, key_findings, analysis = render_claims(notebook, claim_ids)
            summary = "\n\n".join(key_findings)
        elif claim_ids is not None:
            raise ResearchRequestError(
                "Claim-based report completion requires a new evidence-linked job."
            )
        if not summary.strip() or not key_findings:
            raise ValueError("Provide a summary and evidence-backed findings.")
        plan = research_plan(notebook)
        if (
            notebook.require_review
            and not plan["ready_to_complete"]
            and not allow_incomplete
        ):
            raise ResearchRequestError(
                "Research gaps remain. Inspect research_plan, review the evidence, or set allow_incomplete=true to disclose the gaps.",
                code="research_gaps",
            )
        sources = notebook.report_sources()
        if not sources and not (
            notebook.claim_contract_version >= 2 and allow_incomplete
        ):
            raise ValueError("Cannot complete a report without substantive sources.")
        for text in (
            [] if notebook.claim_contract_version >= 2 else [summary, *key_findings]
        ):
            if not re.search(r"\[Source\s+\d+\]", text, re.IGNORECASE):
                raise ValueError(
                    "Summary and each finding must cite a selected [Source N]."
                )
        report = ResearchReport(
            job["question"],
            summary,
            key_findings,
            sources,
            analysis=analysis,
            model_used="host",
            search_queries=[item["query"] for item in notebook.attempts],
            sub_questions=notebook.sub_questions,
            limitations=[
                *notebook.limitations(),
                *limitations,
                "Host-authored report. Citation targets checked; semantic support was not independently verified.",
            ],
            verification_scope="citation_targets_only",
            schema_version="1.1" if report_claims else "1.0",
            claims=report_claims,
            research_stats={
                "retrieval_attempts": len(notebook.attempts),
                "steps": notebook.attempts,
            },
            research_plan=dict(plan),
            status="partial"
            if notebook.require_review and not plan["ready_to_complete"]
            else "ok",
        )
        if citation_errors(report):
            raise ValueError("Report refers to an unknown source citation.")
        job["report"] = report.to_dict()
        job["status"] = "completed"
        self._event(
            job,
            "Host report saved; citation targets checked, semantic support unverified.",
        )
        self.store.save(job)
        return self.get(job_id)

    async def cancel(self, job_id: str, delete: bool = False) -> dict[str, Any]:
        job = self.store.get(job_id)
        task = self.tasks.get(job_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            job = self.store.get(job_id)
        if delete:
            self.store.delete(job_id)
            return {"job_id": job_id, "status": "deleted"}
        if job["status"] != "completed":
            job["status"] = "cancelled"
            self._event(job, "Research cancelled by the caller.")
            self.store.save(job)
        return self.get(job_id)

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.store.close()
