"""Lexical window selection with exact offsets into extracted source text."""

from __future__ import annotations

import re


def relevant_window_span(query: str, text: str, width: int = 6000) -> tuple[int, int]:
    text = text or ""
    if width <= 0:
        return 0, 0
    if len(text) <= width:
        return 0, len(text)
    q_words = {w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    if not q_words:
        return 0, width
    step = max(500, width // 4)
    best_start, best_score = 0, -1
    last_start = len(text) - width
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        chunk = text[start : start + width].lower()
        score = sum(1 for w in q_words if w in chunk)
        if score > best_score:
            best_score, best_start = score, start
    # Prefer the head on ties — intros carry key facts and read cleaner.
    if sum(1 for w in q_words if w in text[:width].lower()) >= best_score:
        best_start = 0
    return best_start, best_start + width


def relevant_window(query: str, text: str, width: int = 6000) -> str:
    """Return the ~width-char slice of a long page with the highest query-keyword
    density — so the answer region of a long article (a tail History table, a deep
    paragraph) is returned instead of just the page head. Falls back to the head
    for short text or a query with no ASCII tokens."""
    text = text or ""
    best_start, end = relevant_window_span(query, text, width)
    prefix = "" if best_start == 0 else "…"
    return prefix + text[best_start + len(prefix) : end]
