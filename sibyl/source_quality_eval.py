"""Offline evaluation helpers for contextual source-quality scorers."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class SourceQualityCandidate:
    candidate_id: str
    title: str
    url: str
    source_type: str


@dataclass(frozen=True)
class SourceQualityCase:
    case_id: str
    query: str
    candidates: List[SourceQualityCandidate]
    preferred_candidate_ids: List[str]
    label_reason: str


@dataclass(frozen=True)
class SourceQualityCaseResult:
    case_id: str
    selected_candidate_id: Optional[str]
    status: str


@dataclass(frozen=True)
class SourceQualityEvalResult:
    total_cases: int
    assessed_cases: int
    correct_cases: int
    coverage: float
    selective_accuracy: float
    overall_accuracy: float
    cases: List[SourceQualityCaseResult]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


QualityScoreFunction = Callable[
    [str, Sequence[SourceQualityCandidate]],
    Sequence[Optional[float]],
]


def _nonempty_string(value: object, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {field} must be a non-empty string.")
    return value.strip()


def _parse_case(data: object, line_number: int) -> SourceQualityCase:
    if not isinstance(data, dict):
        raise ValueError(f"Line {line_number}: each case must be a JSON object.")

    case_id = _nonempty_string(data.get("id"), "id", line_number)
    query = _nonempty_string(data.get("query"), "query", line_number)
    label_reason = _nonempty_string(
        data.get("label_reason"), "label_reason", line_number
    )
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
        raise ValueError(
            f"Line {line_number}: candidates must contain at least two items."
        )

    candidates = []
    candidate_ids = set()
    for index, raw_candidate in enumerate(raw_candidates, 1):
        if not isinstance(raw_candidate, dict):
            raise ValueError(
                f"Line {line_number}: candidate {index} must be a JSON object."
            )
        candidate_id = _nonempty_string(
            raw_candidate.get("id"), f"candidates[{index}].id", line_number
        )
        if candidate_id in candidate_ids:
            raise ValueError(
                f"Line {line_number}: duplicate candidate id {candidate_id!r}."
            )
        candidates.append(
            SourceQualityCandidate(
                candidate_id=candidate_id,
                title=_nonempty_string(
                    raw_candidate.get("title"),
                    f"candidates[{index}].title",
                    line_number,
                ),
                url=_nonempty_string(
                    raw_candidate.get("url"),
                    f"candidates[{index}].url",
                    line_number,
                ),
                source_type=_nonempty_string(
                    raw_candidate.get("source_type"),
                    f"candidates[{index}].source_type",
                    line_number,
                ).lower(),
            )
        )
        candidate_ids.add(candidate_id)

    raw_preferred = data.get("preferred_candidate_ids")
    if not isinstance(raw_preferred, list) or not raw_preferred:
        raise ValueError(
            f"Line {line_number}: preferred_candidate_ids must be a non-empty list."
        )
    preferred_candidate_ids = [
        _nonempty_string(value, "preferred_candidate_ids[]", line_number)
        for value in raw_preferred
    ]
    if len(preferred_candidate_ids) != len(set(preferred_candidate_ids)):
        raise ValueError(
            f"Line {line_number}: preferred candidate ids must be unique."
        )
    unknown = set(preferred_candidate_ids) - candidate_ids
    if unknown:
        raise ValueError(
            f"Line {line_number}: unknown preferred candidate ids: {sorted(unknown)}."
        )

    return SourceQualityCase(
        case_id=case_id,
        query=query,
        candidates=candidates,
        preferred_candidate_ids=preferred_candidate_ids,
        label_reason=label_reason,
    )


def load_source_quality_cases(path: Path) -> List[SourceQualityCase]:
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
        raise ValueError("Source-quality evaluation dataset contains no cases.")
    return cases


def evaluate_source_quality_cases(
    cases: Sequence[SourceQualityCase],
    scorer: QualityScoreFunction,
) -> SourceQualityEvalResult:
    if not cases:
        raise ValueError("At least one source-quality case is required.")

    results = []
    assessed_cases = 0
    correct_cases = 0
    for case in cases:
        raw_scores = scorer(case.query, case.candidates)
        if len(raw_scores) != len(case.candidates):
            raise ValueError(
                f"Case {case.case_id!r}: scorer returned {len(raw_scores)} scores "
                f"for {len(case.candidates)} candidates."
            )

        scores = []
        for score in raw_scores:
            if score is None:
                scores.append(None)
                continue
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(
                    f"Case {case.case_id!r}: scorer returned a non-finite score."
                )
            scores.append(numeric_score)

        available = [score for score in scores if score is not None]
        if not available:
            results.append(SourceQualityCaseResult(case.case_id, None, "abstained"))
            continue
        top_score = max(available)
        top_indices = [
            index for index, score in enumerate(scores) if score == top_score
        ]
        if len(top_indices) != 1:
            results.append(SourceQualityCaseResult(case.case_id, None, "abstained"))
            continue

        assessed_cases += 1
        selected_id = case.candidates[top_indices[0]].candidate_id
        if selected_id in set(case.preferred_candidate_ids):
            correct_cases += 1
            status = "correct"
        else:
            status = "incorrect"
        results.append(SourceQualityCaseResult(case.case_id, selected_id, status))

    total_cases = len(cases)
    return SourceQualityEvalResult(
        total_cases=total_cases,
        assessed_cases=assessed_cases,
        correct_cases=correct_cases,
        coverage=round(assessed_cases / total_cases, 6),
        selective_accuracy=(
            round(correct_cases / assessed_cases, 6) if assessed_cases else 0.0
        ),
        overall_accuracy=round(correct_cases / total_cases, 6),
        cases=results,
    )


_SOURCE_TYPE_PRIORS = {
    "academic": 0.8,
    "wikipedia": 0.65,
    "news": 0.6,
    "web": 0.5,
    "reddit": 0.4,
}


def source_type_prior_scores(
    query: str,
    candidates: Sequence[SourceQualityCandidate],
) -> Sequence[Optional[float]]:
    del query
    return [_SOURCE_TYPE_PRIORS.get(candidate.source_type) for candidate in candidates]
