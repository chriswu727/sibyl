"""Provider-independent research checkpoints and evidence selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .dedup import canonical_url
from .evidence import BundleDiagnostics, EvidencePassage, EvidenceSource, SourceBundle
from .report_models import Source
from .scraper import WebPage
from .source_policy import source_kind, source_priority
from .temporal import period_match


class ResearchBudgetExceeded(RuntimeError):
    pass


@dataclass
class ResearchNotebook:
    question: str
    sub_questions: list[str] = field(default_factory=list)
    bundles: list[SourceBundle] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    max_steps: int = 8
    model_calls: int = 0
    reserved_output_tokens: int = 0
    max_model_calls: int = 40
    max_output_tokens: int = 60000
    synthesis_limit: int = 12
    reflection_rounds: int = 0
    source_policy: str = "primary"
    preferred_domains: list[str] = field(default_factory=list)
    reviews: dict = field(default_factory=dict)
    require_review: bool = False

    def begin_search(self, query: str) -> None:
        if len(self.attempts) >= self.max_steps:
            raise ResearchBudgetExceeded("Research retrieval budget exhausted.")
        self.attempts.append({"query": query, "status": "running", "round": len(self.attempts) + 1})

    def finish_search(self, query: str, bundle: SourceBundle | None) -> None:
        attempt = next(
            item for item in reversed(self.attempts)
            if item["query"] == query and item["status"] == "running"
        )
        attempt["status"] = bundle.status if bundle else "failed"
        attempt["substantive_sources"] = sum(
            s.content_origin not in {"search_snippet", "crossref_api"} and bool(s.evidence)
            for s in bundle.sources
        ) if bundle else 0
        attempt["bundle_id"] = bundle.bundle_id if bundle else ""
        if bundle and bundle.error:
            attempt["error"] = bundle.error
        if bundle is not None:
            self.bundles.append(bundle)

    def reserve_model_call(self, tokens: int) -> None:
        if (self.model_calls >= self.max_model_calls or
                self.reserved_output_tokens + tokens > self.max_output_tokens):
            raise ResearchBudgetExceeded("Research model budget exhausted.")
        self.model_calls += 1
        self.reserved_output_tokens += tokens

    def selected_sources(self) -> list[EvidenceSource]:
        # Interleave query results, newest first: gap-filling sources must not be
        # crowded out by an already-full first search batch.
        groups = [
            sorted([source for source in bundle.sources
                    if source.content_origin != "search_snippet" and source.evidence],
                   key=lambda source: source_priority(source, self.source_policy, self.preferred_domains),
                   reverse=True)
            for bundle in reversed(self.bundles)
            if bundle.status in {"ok", "insufficient_evidence"}
        ]
        selected, urls, clusters = [], set(), set()
        pinned = {ref for review in self.reviews.values()
                  for ref in [*review["supporting_passage_ids"], *review["challenging_passage_ids"]]}

        def add(source):
            url = canonical_url(source.url)
            cluster = source.content_cluster_id or source.content_hash or url
            if url not in urls and cluster not in clusters and len(selected) < self.synthesis_limit:
                selected.append(source)
                urls.add(url)
                clusters.add(cluster)

        for group in groups:
            for source in group:
                if any(p.citation_id in pinned for p in source.evidence):
                    add(source)
        # Background material must not win a round-robin slot ahead of primary leads.
        for background in (False, True):
            for period in (1, 0, -1):
                filtered = [[s for s in group if (source_kind(s, self.preferred_domains) in {
                    "encyclopedia", "community", "scholarly_metadata"
                }) == background and period_match(self.question, s.title, s.url, s.published_at) == period]
                            for group in groups]
                for position in range(max((len(g) for g in filtered), default=0)):
                    for group in filtered:
                        if position < len(group):
                            add(group[position])
        return selected

    def pages(self) -> list[WebPage]:
        return [WebPage(
            source.url, source.title,
            "\n\n".join(passage.text for passage in source.evidence),
            content_origin=source.content_origin,
            published_at=source.published_at,
            published_at_method=source.published_at_method,
            source_type=source.source_type,
        ) for source in self.selected_sources()]

    def has_ready_evidence(self) -> bool:
        return any(
            bundle.status == "ok" and bundle.diagnostics.recommended_action == "synthesize"
            for bundle in self.bundles
        )

    def report_sources(self) -> list[Source]:
        return [Source(
            url=source.url, title=source.title,
            snippet=source.evidence[0].text,
            supporting_snippet=source.evidence[0].text,
            source_id=source.source_id,
            passage_ids=[passage.citation_id for passage in source.evidence],
            retrieved_at=source.retrieved_at, published_at=source.published_at,
            content_origin=source.content_origin,
            content_hash=source.content_hash,
            source_kind=source_kind(source, self.preferred_domains),
            evidence=[asdict(passage) for passage in source.evidence],
        ) for source in self.selected_sources()]

    def limitations(self) -> list[str]:
        from .research_review import question_attempts, questions, research_plan

        notes = ["Public-web research is bounded; inaccessible or unindexed material may be missing."]
        for question_id, question in questions(self).items():
            investigated = any(a["status"] in {"ok", "insufficient_evidence"}
                               for a in question_attempts(self, question_id, question))
            if self.sub_questions and not investigated and not any(bundle.query == question for bundle in self.bundles):
                notes.append(f"Not separately searched: {question}")
        for bundle in self.bundles:
            notes.extend(note for note in bundle.diagnostics.sufficiency_reasons if note.startswith("PDF "))
            if bundle.diagnostics.evidence_sufficiency != "sufficient":
                notes.append(f"Retrieval coverage needs review for: {bundle.query}")
        if len(self.attempts) >= self.max_steps:
            notes.append("Retrieval step budget reached; additional gaps may remain.")
        if any(item["status"] in {"failed", "running"} for item in self.attempts):
            notes.append("Some retrieval attempts failed or were interrupted.")
        if self.require_review:
            for item in research_plan(self)["questions"]:
                notes.extend(f"{item['question_id']}: {gap}" for gap in item["gaps"])
        if any(source_kind(s, self.preferred_domains) == "preprint" for s in self.selected_sources()):
            notes.append("Preprints are not assumed peer reviewed; inspect publication and replication status.")
        notes.append("Source-type and preferred-domain labels are routing hints, not verified authority or truth.")
        return list(dict.fromkeys(notes))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ResearchNotebook:
        values = dict(data)
        values["bundles"] = [SourceBundle(
            **{key: value for key, value in bundle.items()
               if key not in {"sources", "diagnostics"}},
            sources=[EvidenceSource(
                **{key: value for key, value in source.items() if key != "evidence"},
                evidence=[EvidencePassage(**passage) for passage in source["evidence"]],
            ) for source in bundle["sources"]],
            diagnostics=BundleDiagnostics(**bundle["diagnostics"]),
        ) for bundle in values.get("bundles", [])]
        return cls(**values)
