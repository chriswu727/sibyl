"""Deterministic query constraints, source identity, and evidence readiness policy."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from .dedup import canonical_url
from .evidence import EvidenceSufficiency
from .queries import historical_role_requirement
from .ranking import lexical_query_terms

_QUERY_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_QUOTED_TEXT_RE = re.compile(r"""["“]([^"”]+)["”]|(?<!\w)'([^']{3,})'(?!\w)""")
_QUESTION_OPENERS = {
    "after",
    "at",
    "how",
    "in",
    "on",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
}
_FUTURE_YEAR_RE = re.compile(r"\b[2-9]\d{3}\b")
_TENURE_RANGE_RE = re.compile(
    r"\b([12]\d{3})\s*(?:-|–|—|to|through|until)\s*([12]\d{3})\b",
    re.IGNORECASE,
)
_ROLE_MODIFIERS = {
    "acting",
    "assistant",
    "deputy",
    "former",
    "honorary",
    "interim",
    "vice",
}


def _query_anchor_terms(query: str) -> set[str]:
    anchors = set()
    words = _QUERY_WORD_RE.findall(query)
    for index, word in enumerate(words):
        if len(word) < 4 or not word[0].isupper():
            continue
        if index == 0 and word.casefold() in _QUESTION_OPENERS:
            continue
        anchors.update(lexical_query_terms(word))
    for match in _QUOTED_TEXT_RE.finditer(query):
        anchors.update(lexical_query_terms(match.group(1) or match.group(2)))
    return anchors


def _entity_lookup_queries(query: str) -> List[str]:
    quoted = []
    for match in _QUOTED_TEXT_RE.finditer(query):
        value = " ".join((match.group(1) or match.group(2)).split())
        if value:
            quoted.append(value)
    if quoted:
        return quoted[:2]

    sequences = []
    current = []
    for index, word in enumerate(_QUERY_WORD_RE.findall(query)):
        capitalized = len(word) >= 2 and word[0].isupper()
        is_opener = index == 0 and word.casefold() in _QUESTION_OPENERS
        if capitalized and not is_opener:
            current.append(word)
        elif current:
            sequences.append(" ".join(current))
            current = []
    if current:
        sequences.append(" ".join(current))
    return sequences[:2]


def _future_outcome_is_unobservable(query: str) -> bool:
    if " will " not in f" {query.casefold()} ":
        return False
    current_year = datetime.now(timezone.utc).year
    return any(int(year) > current_year for year in _FUTURE_YEAR_RE.findall(query))


def _has_historical_role_support(query: str, evidence_units: List[str]) -> bool:
    requirement = historical_role_requirement(query)
    if requirement is None:
        return True
    role, target_year = requirement
    role_pattern = re.compile(rf"\b{re.escape(role)}\b", re.IGNORECASE)
    target_pattern = re.compile(rf"\b{target_year}\b")
    for unit in evidence_units:
        for statement in re.split(r"(?<=[.!?])\s+", unit):
            for role_match in role_pattern.finditer(statement):
                prefix = statement[max(0, role_match.start() - 20) : role_match.start()]
                prefix_terms = _QUERY_WORD_RE.findall(prefix.casefold())
                if prefix_terms and prefix_terms[-1] in _ROLE_MODIFIERS:
                    continue
                if any(
                    abs(year_match.start() - role_match.start()) <= 180
                    for year_match in target_pattern.finditer(statement)
                ):
                    return True
                for range_match in _TENURE_RANGE_RE.finditer(statement):
                    if abs(range_match.start() - role_match.start()) > 180:
                        continue
                    start_year, end_year = map(int, range_match.groups())
                    if start_year <= target_year <= end_year:
                        return True
    return False


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bundle_id(
    query: str, status: str, source_fingerprints: List[Tuple[str, str, str]]
) -> str:
    payload = json.dumps(
        {"query": query.strip(), "status": status, "sources": source_fingerprints},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"sb_{_sha256(payload)[:16]}"


def _source_type(url: str, result_types: Dict[str, str]) -> str:
    if "wikipedia.org/wiki/" in url:
        return "wikipedia"
    return result_types.get(canonical_url(url), "web")


def _assess_evidence_sufficiency(
    *,
    source_count: int,
    substantive_sources: int,
    evidence_chars: int,
    query_terms: int,
    query_term_coverage: float,
    unique_domains: int,
    independent_content_clusters: int,
    missing_anchor_terms: List[str],
    future_outcome_unobservable: bool,
    fragmented_query_support: bool,
    multi_step_query: bool,
    historical_role_support_missing: bool,
) -> Tuple[EvidenceSufficiency, List[str]]:
    if source_count == 0:
        return "insufficient", ["no_sources"]

    blockers = []
    if substantive_sources == 0:
        blockers.append("no_substantive_sources")
    if evidence_chars < 200:
        blockers.append("too_little_evidence_text")
    if query_terms > 0 and query_term_coverage < 0.25:
        blockers.append("low_query_term_coverage")
    if missing_anchor_terms:
        blockers.append("missing_query_anchor")
    if future_outcome_unobservable:
        blockers.append("future_outcome_not_observable")
    if fragmented_query_support:
        blockers.append("fragmented_query_support")
    if multi_step_query:
        blockers.append("multi_step_query")
    if historical_role_support_missing:
        blockers.append("missing_historical_role_tenure")
    if blockers:
        return "insufficient", blockers

    limitations = []
    if substantive_sources < 2:
        limitations.append("fewer_than_two_substantive_sources")
    elif independent_content_clusters < 2:
        limitations.append("fewer_than_two_independent_contents")
    if unique_domains < 2:
        limitations.append("single_domain")
    if query_terms == 0:
        limitations.append("query_has_no_lexical_terms")
    if limitations:
        return "limited", limitations
    return "sufficient", []


_SUFFICIENCY_REASON_LABELS = {
    "no_sources": "no sources",
    "no_substantive_sources": "no substantive full-text sources",
    "too_little_evidence_text": "too little evidence text",
    "low_query_term_coverage": "low query-term coverage",
    "missing_query_anchor": "one or more key query entities are absent",
    "future_outcome_not_observable": "the requested future outcome is not yet observable",
    "fragmented_query_support": (
        "no single source sufficiently covers the quoted target and requested fact"
    ),
    "multi_step_query": "the question contains a dependent fact chain",
    "missing_historical_role_tenure": (
        "no local statement connects the requested role to the target year"
    ),
}


def _insufficient_evidence_error(
    query: str,
    source_count: int,
    reasons: List[str],
) -> str:
    if source_count == 0:
        return f"No sources found for query: {query!r}. Try a different phrasing."
    details = ", ".join(
        _SUFFICIENCY_REASON_LABELS.get(reason, reason.replace("_", " "))
        for reason in reasons
    )
    return (
        f"Found {source_count} source(s), but the evidence is insufficient for "
        f"synthesis ({details}). Treat these as leads and gather more evidence."
    )
