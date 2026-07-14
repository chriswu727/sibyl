"""Offline end-to-end evaluation for the structured retrieval pipeline."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence
from unittest import mock

from .ranking import RankingBackend
from .retrieval import gather_source_bundle
from .retrieval_eval import RetrievalCase, RetrievalDocument
from .scraper import WebPage
from .search import SearchResult


@dataclass(frozen=True)
class RetrievalPipelineCaseResult:
    case_id: str
    top_document_id: str
    top_source_hit: bool
    status: str
    evidence_sufficiency: str
    structure_valid: bool

    @property
    def passed(self) -> bool:
        return (
            self.top_source_hit
            and self.status == "ok"
            and self.evidence_sufficiency in {"sufficient", "limited"}
            and self.structure_valid
        )


@dataclass(frozen=True)
class RetrievalPipelineEvalResult:
    total_cases: int
    top_source_hits: int
    top_source_accuracy: float
    usable_bundles: int
    usable_bundle_rate: float
    structurally_valid_bundles: int
    structure_valid_rate: float
    passed_cases: int
    case_pass_rate: float
    cases: List[RetrievalPipelineCaseResult]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        for case, result in zip(data["cases"], self.cases):
            case["passed"] = result.passed
        return data


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _substantive_text(document: RetrievalDocument) -> str:
    base = ". ".join(part.strip() for part in [document.title, document.text] if part.strip())
    repetitions = max(1, math.ceil(320 / len(base)))
    return " ".join([base] * repetitions)


def _bundle_structure_is_valid(bundle, page_text_by_url: Dict[str, str]) -> bool:
    if bundle.schema_version != "1.5" or not bundle.sources:
        return False
    for source_index, source in enumerate(bundle.sources, 1):
        page_text = page_text_by_url.get(source.url)
        if page_text is None:
            return False
        if source.source_id != f"S{source_index}":
            return False
        if source.content_hash != hashlib.sha256(page_text.encode("utf-8")).hexdigest():
            return False
        if not source.evidence:
            return False
        for passage_index, passage in enumerate(source.evidence, 1):
            if passage.passage_id != f"P{passage_index}":
                return False
            if passage.citation_id != (
                f"{bundle.bundle_id}/{source.source_id}/{passage.passage_id}"
            ):
                return False
            if page_text[passage.start_char:passage.end_char] != passage.text:
                return False
            if passage.content_hash != hashlib.sha256(
                passage.text.encode("utf-8")
            ).hexdigest():
                return False
    return True


async def evaluate_pipeline_case(
    case: RetrievalCase,
    ranker: RankingBackend = "lexical",
) -> RetrievalPipelineCaseResult:
    results = []
    pages = []
    document_id_by_url = {}
    for document in case.documents:
        url = (
            f"https://{_slug(document.document_id)}."
            f"{_slug(case.case_id)}.eval.invalid/source"
        )
        results.append(
            SearchResult(document.title, url, document.text[:200], "web")
        )
        pages.append(
            WebPage(url, document.title, _substantive_text(document))
        )
        document_id_by_url[url] = document.document_id
    page_by_url = {page.url: page for page in pages}

    async def scrape_fixture(urls, **kwargs):
        return [page_by_url[url] for url in urls]

    with mock.patch(
        "sibyl.retrieval.search_web",
        new=mock.AsyncMock(return_value=results),
    ), mock.patch(
        "sibyl.retrieval.scrape_urls",
        new=scrape_fixture,
    ), mock.patch(
        "sibyl.retrieval.wikipedia_lookup",
        new=mock.AsyncMock(return_value=[]),
    ):
        bundle = await gather_source_bundle(
            case.query,
            max_sources=min(3, len(case.documents)),
            chars_per_source=2000,
            client=object(),
            ranker=ranker,
        )

    top_document_id = (
        document_id_by_url.get(bundle.sources[0].url, "")
        if bundle.sources
        else ""
    )
    return RetrievalPipelineCaseResult(
        case_id=case.case_id,
        top_document_id=top_document_id,
        top_source_hit=top_document_id in set(case.relevant_document_ids),
        status=bundle.status,
        evidence_sufficiency=bundle.diagnostics.evidence_sufficiency,
        structure_valid=_bundle_structure_is_valid(
            bundle,
            {page.url: page.text for page in pages},
        ),
    )


async def evaluate_pipeline_cases(
    cases: Sequence[RetrievalCase],
    ranker: RankingBackend = "lexical",
) -> RetrievalPipelineEvalResult:
    if not cases:
        raise ValueError("At least one retrieval case is required.")
    results = [
        await evaluate_pipeline_case(case, ranker=ranker)
        for case in cases
    ]
    total = len(results)
    top_hits = sum(result.top_source_hit for result in results)
    usable = sum(
        result.status == "ok"
        and result.evidence_sufficiency in {"sufficient", "limited"}
        for result in results
    )
    valid = sum(result.structure_valid for result in results)
    passed = sum(result.passed for result in results)
    return RetrievalPipelineEvalResult(
        total_cases=total,
        top_source_hits=top_hits,
        top_source_accuracy=round(top_hits / total, 6),
        usable_bundles=usable,
        usable_bundle_rate=round(usable / total, 6),
        structurally_valid_bundles=valid,
        structure_valid_rate=round(valid / total, 6),
        passed_cases=passed,
        case_pass_rate=round(passed / total, 6),
        cases=results,
    )
