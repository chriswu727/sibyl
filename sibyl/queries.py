"""Deterministic search-query variants."""
from __future__ import annotations

import re
from typing import List


_TERM_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_QUESTION_WORDS = {"how", "what", "when", "where", "which", "who", "why"}
_SEARCH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def search_query_variants(query: str) -> List[str]:
    clean = " ".join(str(query or "").split())
    if not clean:
        return []

    terms = _TERM_RE.findall(clean)
    if not terms or terms[0].casefold() not in _QUESTION_WORDS:
        return [clean]

    focused_terms = [
        term
        for term in terms[1:]
        if term.casefold() not in _SEARCH_STOP_WORDS
    ]
    focused = " ".join(focused_terms)
    if len(focused_terms) < 3 or focused.casefold() == clean.casefold():
        return [clean]
    return [clean, focused]
