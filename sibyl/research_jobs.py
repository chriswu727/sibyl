"""Durable host-assisted and model-backed research orchestration."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from .async_cache import CapacityError
from .job_store import JobStore
from .report_audit import citation_errors
from .report_models import ResearchReport
from .research_state import ResearchBudgetExceeded, ResearchNotebook
from .research_errors import ResearchRequestError
from .retrieval import gather_source_bundle
from .privacy import display_url
from .research_review import evidence_index, questions, research_plan, review_question
from .research_sources import read_source_bundle
from .source_policy import normalize_domains, source_kind


class ResearchJobs:
    def __init__(self, runtime):
        self.runtime = runtime
        self.store = JobStore(runtime.policy.output_root)
        self.tasks: dict[str, asyncio.Task] = {}
        self.closed = False

    def _event(self, job, message):
        job["events"] = [*job["events"][-79:], {"message": message[:240], "time": time.time()}]

    def _checkpoint(self, job):
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
                if attempt["status"] == "running" or attempt.get("bundle_id", "") not in retained_ids:
                    attempt.update(status="failed", bundle_id="", substantive_sources=0)
                    attempt.pop("error", None)
            retained["elapsed_seconds"] = job["elapsed_seconds"]
            retained["status"] = "awaiting_input" if job["mode"] == "host" else "failed"
            if job["status"] in {"interrupted", "budget_exhausted"}:
                retained["status"] = job["status"]
            self._event(retained, "Checkpoint size limit reached; new evidence/report not saved. Prior evidence and consumed budgets retained.")
            self.store.save(retained, recovery=True)
            job.clear()
            job.update(retained)
            raise CapacityError("Checkpoint size limit reached; prior evidence and consumed budgets retained.") from None

    def _idle(self, job):
        if self.closed:
            raise ValueError("Research runtime is closed.")
        if job["job_id"] in self.tasks:
            raise CapacityError("Research job is busy.")
        if job["status"] == "completed":
            raise ValueError("Completed reports are immutable. Start a new research job.")

    def start(self, question, sub_questions, mode, depth, language, max_steps,
              source_policy="primary", preferred_domains=None):
        self.runtime.validate_query(question)
        if not question.strip() or mode not in {"host", "autonomous"} or depth not in {1, 2, 3}:
            raise ValueError("Invalid research request.")
        if not 1 <= max_steps <= 12 or len(sub_questions) > 5:
            raise ValueError("Use 1-12 retrieval steps and at most five sub-questions.")
        if source_policy not in {"primary", "academic"}:
            raise ValueError("Source policy must be primary or academic.")
        preferred_domains = normalize_domains(preferred_domains or [])
        for item in sub_questions:
            self.runtime.validate_query(item)
            if not item.strip():
                raise ValueError("Sub-questions must not be empty.")
        if mode == "autonomous":
            if not self.runtime.config.has_llm_credentials():
                raise ValueError("Autonomous research requires a configured model.")
            from .researcher import Researcher  # noqa: F401
            if len(self.tasks) >= 2:
                raise CapacityError("Research worker capacity reached.")
        notebook = ResearchNotebook(
            question, sub_questions=list(dict.fromkeys(sub_questions)), max_steps=max_steps,
            source_policy=source_policy, preferred_domains=preferred_domains,
            require_review=mode == "host",
        )
        job = self.store.create({
            "question": question, "mode": mode, "depth": depth, "language": language,
            "status": "awaiting_input", "notebook": notebook.to_dict(),
            "report": None, "events": [], "elapsed_seconds": 0.0,
        })
        self._event(job, "Research created; sources are untrusted data.")
        self.store.save(job)
        if mode == "autonomous":
            self._launch(job)
        return self.get(job["job_id"])

    def get(self, job_id, passage_id=""):
        job = self.store.get(job_id)
        if job["status"] == "running" and job_id not in self.tasks:
            job["status"] = "interrupted"
        notebook = ResearchNotebook.from_dict(job["notebook"])
        if passage_id:
            item = evidence_index(notebook).get(passage_id)
            if item is None:
                raise ValueError("Unknown evidence passage.")
            source, passage = item
            return dict(job_id=job_id, revision=job["revision"], url=display_url(source.url),
                        title=source.title, passage=passage.__dict__,
                        source_kind=source_kind(source, notebook.preferred_domains))
        public = {key: value for key, value in job.items() if key != "notebook"}
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
            evidence=[source.__dict__ | {"evidence": [p.__dict__ for p in source.evidence]}
                      for source in notebook.selected_sources()],
            limitations=notebook.limitations(),
            research_plan=research_plan(notebook),
            source_policy=notebook.source_policy,
            preferred_domains=notebook.preferred_domains,
            source_catalog=[dict(
                url=display_url(source.url), title=source.title,
                source_kind=source_kind(source, notebook.preferred_domains),
                content_origin=source.content_origin, retrieved_at=source.retrieved_at,
                published_at=source.published_at,
                passage_ids=[p.citation_id for p in source.evidence],
            ) for bundle in notebook.bundles for source in bundle.sources],
        )
        for source in [*public["sources"], *public["evidence"]]:
            source["url"] = display_url(source["url"])
        if public["report"]:
            for source in public["report"]["sources"]:
                source["url"] = display_url(source["url"])
        return public

    def _launch(self, job):
        self._idle(job)
        if len(self.tasks) >= 2:
            raise CapacityError("Research worker capacity reached.")
        if job["elapsed_seconds"] >= self.runtime.policy.report_timeout:
            raise ResearchBudgetExceeded("Research wall-time budget exhausted.")
        job["status"] = "running"
        self._event(job, "Autonomous research running.")
        self.store.save(job)
        task = asyncio.create_task(self._run(job))
        self.tasks[job["job_id"]] = task

        def finished(completed):
            if self.tasks.get(job["job_id"]) is completed:
                self.tasks.pop(job["job_id"], None)
            if not completed.cancelled():
                completed.exception()
        task.add_done_callback(finished)

    async def _run(self, job):
        from .researcher import Researcher

        started = time.monotonic()
        elapsed = job["elapsed_seconds"]
        notebook = ResearchNotebook.from_dict(job["notebook"])

        def checkpoint(state):
            job["notebook"] = state.to_dict()
            job["elapsed_seconds"] = elapsed + time.monotonic() - started
            self._checkpoint(job)

        def progress(message):
            self._event(job, message)
            checkpoint(notebook)

        try:
            config = replace(self.runtime.config, providers=list(self.runtime.config.providers))
            researcher = Researcher(config, notebook=notebook, on_checkpoint=checkpoint)
            report = await asyncio.wait_for(
                researcher.research(
                    job["question"], depth=job["depth"], language=job["language"],
                    on_progress=progress,
                ),
                self.runtime.policy.report_timeout - elapsed,
            )
            job["report"] = report.to_dict()
            job["status"] = "completed" if report.status == "ok" else "failed"
            self._event(job, "Report ready." if report.status == "ok" else "No complete report produced.")
        except asyncio.CancelledError:
            job["status"] = "interrupted"
            self._event(job, "Research interrupted; completed evidence checkpoints retained.")
            raise
        except (asyncio.TimeoutError, ResearchBudgetExceeded):
            job["status"] = "budget_exhausted"
            self._event(job, "Research budget exhausted.")
        except Exception:
            job["status"] = "failed"
            self._event(job, "Research failed. Check model configuration and retry explicitly.")
        finally:
            try:
                checkpoint(notebook)
            finally:
                self.tasks.pop(job["job_id"], None)

    async def continue_job(self, job_id, query, url="", question_id="", purpose="explore", parent_passage_id="",
                           rationale=""):
        job = self.store.get(job_id)
        self._idle(job)
        if job["mode"] == "autonomous":
            if query or url or question_id or parent_passage_id or rationale or purpose != "explore":
                raise ValueError("Autonomous continuation reuses its saved plan; omit query.")
            self._launch(job)
            return self.get(job_id)
        self.runtime.validate_query(query)
        if not query.strip() and not url:
            raise ValueError("Host-assisted research requires a focused query or source URL.")
        if purpose not in {"explore", "counterevidence", "follow_citation"}:
            raise ValueError("Unknown investigation purpose.")
        if len(url) > 2000:
            raise ValueError("Source URL is too long.")
        if len(rationale) > 1000:
            raise ResearchRequestError("Retrieval rationale must be at most 1000 characters.")
        if len(self.tasks) >= 2:
            raise CapacityError("Research worker capacity reached.")
        remaining = self.runtime.policy.report_timeout - job["elapsed_seconds"]
        if remaining <= 0:
            raise ResearchBudgetExceeded("Research wall-time budget exhausted.")
        notebook = ResearchNotebook.from_dict(job["notebook"])
        plan_questions = questions(notebook)
        question_id = question_id or next((key for key, value in plan_questions.items() if value == query), "")
        if not question_id and len(plan_questions) == 1:
            question_id = next(iter(plan_questions))
        if question_id not in plan_questions:
            raise ValueError("Choose a question_id from the research plan.")
        if parent_passage_id and parent_passage_id not in evidence_index(notebook):
            raise ValueError("Parent passage must belong to this investigation.")
        if purpose == "follow_citation" and (not url or not parent_passage_id):
            raise ValueError("Citation following requires a URL and parent passage ID.")
        query = query.strip() or plan_questions[question_id]
        previous_question = next(item for item in research_plan(notebook)["questions"]
                                 if item["question_id"] == question_id)
        notebook.begin_search(query)
        notebook.attempts[-1].update(question_id=question_id, purpose=purpose,
                                     url=display_url(url) if url else "",
                                     parent_passage_id=parent_passage_id,
                                     rationale=rationale.strip(), based_on_revision=job["revision"],
                                     prior_gaps=previous_question["gaps"],
                                     prior_review_status=(previous_question["review"] or {}).get("status"))
        job["notebook"] = notebook.to_dict()
        job["status"] = "running"
        self._event(job, f"Retrieving evidence: {query}")
        self.store.save(job)
        started = time.monotonic()
        self.tasks[job_id] = asyncio.current_task()
        try:
            bundle = await asyncio.wait_for(
                read_source_bundle(url, query) if url else gather_source_bundle(
                    query, max_sources=8, chars_per_source=4000,
                    include_academic=True if notebook.source_policy == "academic" else None,
                ),
                min(remaining, self.runtime.policy.tool_timeout * 0.9),
            )
            notebook.finish_search(query, bundle)
            job["status"] = "awaiting_input"
            self._event(job, f"Retrieval finished: {bundle.status}; inspect evidence before writing.")
        except asyncio.CancelledError:
            notebook.finish_search(query, None)
            job["status"] = "interrupted"
            raise
        except Exception:
            notebook.finish_search(query, None)
            job["status"] = "awaiting_input"
            self._event(job, "Retrieval failed; no evidence inferred. Retry or refine explicitly.")
        finally:
            job["notebook"] = notebook.to_dict()
            job["elapsed_seconds"] += time.monotonic() - started
            try:
                self._checkpoint(job)
            finally:
                self.tasks.pop(job_id, None)
        return self.get(job_id)

    def assess(self, job_id, revision, question_id, status, supporting, challenging, conclusion, counterevidence_note,
               follow_up_queries=None):
        job = self.store.get(job_id)
        self._idle(job)
        if job["mode"] != "host" or revision != job["revision"]:
            raise ResearchRequestError("Reload the host research job and use its current revision.")
        notebook = ResearchNotebook.from_dict(job["notebook"])
        review_question(notebook, question_id, status, supporting, challenging, conclusion, counterevidence_note,
                        follow_up_queries)
        job["notebook"] = notebook.to_dict()
        self._event(job, f"Host reviewed {question_id}: {status}; semantic support is not independently verified.")
        self.store.save(job)
        return self.get(job_id)

    def complete(self, job_id, revision, summary, key_findings, analysis, limitations, allow_incomplete=False):
        import re

        job = self.store.get(job_id)
        self._idle(job)
        if job["mode"] != "host" or job["revision"] != revision:
            raise ResearchRequestError("Reload the host research job and use its current revision.")
        if not summary.strip() or not key_findings:
            raise ValueError("Provide a summary and evidence-backed findings.")
        if len(key_findings) > 32 or len(limitations) > 32:
            raise ValueError("At most 32 findings and limitations are supported.")
        notebook = ResearchNotebook.from_dict(job["notebook"])
        plan = research_plan(notebook)
        if notebook.require_review and not plan["ready_to_complete"] and not allow_incomplete:
            raise ResearchRequestError("Research gaps remain. Inspect research_plan, review the evidence, or set allow_incomplete=true to disclose the gaps.")
        sources = notebook.report_sources()
        if not sources:
            raise ValueError("Cannot complete a report without substantive sources.")
        for text in [summary, *key_findings]:
            if not re.search(r"\[Source\s+\d+\]", text, re.IGNORECASE):
                raise ValueError("Summary and each finding must cite a selected [Source N].")
        report = ResearchReport(
            job["question"], summary, key_findings, sources, analysis=analysis,
            model_used="host", search_queries=[item["query"] for item in notebook.attempts],
            sub_questions=notebook.sub_questions,
            limitations=[*notebook.limitations(), *limitations,
                         "Host-authored report. Citation targets checked; semantic support was not independently verified."],
            verification_scope="citation_targets_only",
            research_stats={"retrieval_attempts": len(notebook.attempts), "steps": notebook.attempts},
            research_plan=plan,
            status="partial" if notebook.require_review and not plan["ready_to_complete"] else "ok",
        )
        if citation_errors(report):
            raise ValueError("Report refers to an unknown source citation.")
        job["report"] = report.to_dict()
        job["status"] = "completed"
        self._event(job, "Host report saved; citation targets checked, semantic support unverified.")
        self.store.save(job)
        return self.get(job_id)

    async def cancel(self, job_id, delete=False):
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

    async def close(self):
        self.closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.store.close()
