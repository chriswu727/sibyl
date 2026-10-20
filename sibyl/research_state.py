"""Provider-independent research checkpoints and evidence selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .dedup import canonical_url
from .evidence import BundleDiagnostics, EvidencePassage, EvidenceSource, SourceBundle
from .report_models import Source
from .scraper import WebPage


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

    def begin_search(self, query: str) -> None:
        if len(self.attempts) >= self.max_steps:
            raise ResearchBudgetExceeded("Research retrieval budget exhausted.")
        self.attempts.append({"query": query, "status": "running"})

    def finish_search(self, query: str, bundle: SourceBundle | None) -> None:
        attempt = next(
            item for item in reversed(self.attempts)
            if item["query"] == query and item["status"] == "running"
        )
        attempt["status"] = bundle.status if bundle else "failed"
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
            [source for source in bundle.sources
             if source.content_origin != "search_snippet" and source.evidence]
            for bundle in reversed(self.bundles)
            if bundle.status in {"ok", "insufficient_evidence"}
        ]
        selected, urls, clusters = [], set(), set()
        for position in range(max((len(group) for group in groups), default=0)):
            for group in groups:
                if position >= len(group):
                    continue
                source = group[position]
                url = canonical_url(source.url)
                cluster = source.content_cluster_id or source.content_hash
                if url in urls or cluster in clusters:
                    continue
                selected.append(source)
                urls.add(url)
                clusters.add(cluster)
                if len(selected) >= self.synthesis_limit:
                    return selected
        return selected

    def pages(self) -> list[WebPage]:
        return [WebPage(
            source.url, source.title,
            "\n\n".join(passage.text for passage in source.evidence),
            content_origin=source.content_origin,
            published_at=source.published_at,
            published_at_method=source.published_at_method,
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
        ) for source in self.selected_sources()]

    def limitations(self) -> list[str]:
        notes = ["Public-web research is bounded; inaccessible or unindexed material may be missing."]
        for question in self.sub_questions:
            if not any(bundle.query == question for bundle in self.bundles):
                notes.append(f"Not separately searched: {question}")
        for bundle in self.bundles:
            if bundle.diagnostics.evidence_sufficiency != "sufficient":
                notes.append(f"Retrieval coverage needs review for: {bundle.query}")
        if len(self.attempts) >= self.max_steps:
            notes.append("Retrieval step budget reached; additional gaps may remain.")
        if any(item["status"] in {"failed", "running"} for item in self.attempts):
            notes.append("Some retrieval attempts failed or were interrupted.")
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
