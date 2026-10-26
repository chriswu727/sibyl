"""Lexical window selection with exact offsets into extracted source text."""

from __future__ import annotations

from .passages import TextPassage, passage_relevance_scores
from .ranking import lexical_query_terms


def relevant_window_span(query: str, text: str, width: int = 6000) -> tuple[int, int]:
    text = text or ""
    if width <= 0:
        return 0, 0
    if len(text) <= width:
        return 0, len(text)
    if not lexical_query_terms(query):
        return 0, width
    step = max(500, width // 4)
    last_start = len(text) - width
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    scores = passage_relevance_scores(
        query,
        [
            TextPassage(text[start : start + width], start, start + width)
            for start in starts
        ],
    )
    best_start = starts[
        max(range(len(starts)), key=lambda index: (scores[index], -index))
    ]
    return best_start, best_start + width


def relevant_window(query: str, text: str, width: int = 6000) -> str:
    """Return a bounded lexical window, preferring body matches over navigation.

    Falls back to the head for short text or a query with no lexical terms.
    """
    text = text or ""
    best_start, end = relevant_window_span(query, text, width)
    prefix = "" if best_start == 0 else "…"
    return prefix + text[best_start + len(prefix) : end]
