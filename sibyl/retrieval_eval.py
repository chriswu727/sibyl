"""Offline evaluation helpers for retrieval rankers."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple


ScoreFunction = Callable[[str, Sequence[Tuple[str, str]]], List[float]]


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    documents: List[RetrievalDocument]
    relevant_document_ids: List[str]


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    top_document_id: str
    first_relevant_rank: int


@dataclass(frozen=True)
class RetrievalEvalResult:
    total_cases: int
    hits_at_1: int
    hit_at_1: float
    mean_reciprocal_rank: float
    cases: List[RetrievalCaseResult]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _nonempty_string(value: object, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {field} must be a non-empty string.")
    return value.strip()


def _parse_case(data: object, line_number: int) -> RetrievalCase:
    if not isinstance(data, dict):
        raise ValueError(f"Line {line_number}: each case must be a JSON object.")

    case_id = _nonempty_string(data.get("id"), "id", line_number)
    query = _nonempty_string(data.get("query"), "query", line_number)
    raw_documents = data.get("documents")
    if not isinstance(raw_documents, list) or len(raw_documents) < 2:
        raise ValueError(f"Line {line_number}: documents must contain at least two items.")

    documents = []
    document_ids = set()
    for index, raw_document in enumerate(raw_documents, 1):
        if not isinstance(raw_document, dict):
            raise ValueError(
                f"Line {line_number}: document {index} must be a JSON object."
            )
        document_id = _nonempty_string(
            raw_document.get("id"), f"documents[{index}].id", line_number
        )
        if document_id in document_ids:
            raise ValueError(
                f"Line {line_number}: duplicate document id {document_id!r}."
            )
        title = raw_document.get("title", "")
        text = raw_document.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(
                f"Line {line_number}: document {document_id!r} title/text must be strings."
            )
        if not title.strip() and not text.strip():
            raise ValueError(
                f"Line {line_number}: document {document_id!r} must contain title or text."
            )
        documents.append(RetrievalDocument(document_id, title, text))
        document_ids.add(document_id)

    raw_relevant = data.get("relevant_document_ids")
    if not isinstance(raw_relevant, list) or not raw_relevant:
        raise ValueError(
            f"Line {line_number}: relevant_document_ids must be a non-empty list."
        )
    relevant_document_ids = [
        _nonempty_string(value, "relevant_document_ids[]", line_number)
        for value in raw_relevant
    ]
    unknown = set(relevant_document_ids) - document_ids
    if unknown:
        raise ValueError(
            f"Line {line_number}: unknown relevant document ids: {sorted(unknown)}."
        )
    if len(relevant_document_ids) != len(set(relevant_document_ids)):
        raise ValueError(f"Line {line_number}: relevant document ids must be unique.")

    return RetrievalCase(case_id, query, documents, relevant_document_ids)


def load_retrieval_cases(path: Path) -> List[RetrievalCase]:
    cases = []
    case_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc
        case = _parse_case(data, line_number)
        if case.case_id in case_ids:
            raise ValueError(f"Line {line_number}: duplicate case id {case.case_id!r}.")
        cases.append(case)
        case_ids.add(case.case_id)
    if not cases:
        raise ValueError("Retrieval evaluation dataset contains no cases.")
    return cases


def evaluate_retrieval_cases(
    cases: Sequence[RetrievalCase],
    scorer: ScoreFunction,
) -> RetrievalEvalResult:
    if not cases:
        raise ValueError("At least one retrieval case is required.")

    results = []
    reciprocal_rank_sum = 0.0
    hits_at_1 = 0
    for case in cases:
        inputs = [(document.title, document.text) for document in case.documents]
        raw_scores = scorer(case.query, inputs)
        if len(raw_scores) != len(case.documents):
            raise ValueError(
                f"Case {case.case_id!r}: scorer returned {len(raw_scores)} scores "
                f"for {len(case.documents)} documents."
            )
        scores = []
        for score in raw_scores:
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(
                    f"Case {case.case_id!r}: scorer returned a non-finite score."
                )
            scores.append(numeric_score)

        ranked_indices = sorted(
            range(len(case.documents)), key=lambda index: (-scores[index], index)
        )
        relevant = set(case.relevant_document_ids)
        first_relevant_rank = next(
            rank
            for rank, index in enumerate(ranked_indices, 1)
            if case.documents[index].document_id in relevant
        )
        top_document_id = case.documents[ranked_indices[0]].document_id
        if first_relevant_rank == 1:
            hits_at_1 += 1
        reciprocal_rank_sum += 1 / first_relevant_rank
        results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                top_document_id=top_document_id,
                first_relevant_rank=first_relevant_rank,
            )
        )

    total_cases = len(cases)
    return RetrievalEvalResult(
        total_cases=total_cases,
        hits_at_1=hits_at_1,
        hit_at_1=round(hits_at_1 / total_cases, 6),
        mean_reciprocal_rank=round(reciprocal_rank_sum / total_cases, 6),
        cases=results,
    )
