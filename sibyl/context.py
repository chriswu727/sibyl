"""Shared helpers for building LLM source context and picking supporting snippets.

Kept in its own module so both researcher.py and verifier.py can use them
without an import cycle. The [Source N] numbering is defined here: whatever page
order build_source_context receives IS the citation order the synthesis prompt,
the verifier, and the rendered reference list must all key off.
"""
from __future__ import annotations

import re
from typing import List

from .scraper import WebPage


def build_source_context(pages: List[WebPage], limit: int = 12, per_char: int = 4000) -> str:
    """Render pages into the numbered [Source N] context block used by synthesis
    and verification. Page i (1-based) becomes [Source i]."""
    parts = []
    for i, page in enumerate(pages[:limit], 1):
        parts.append(f"[Source {i}: {page.title}]\nURL: {page.url}\n{page.text[:per_char]}\n")
    return "\n---\n".join(parts)


def relevant_window(query: str, text: str, width: int = 6000) -> str:
    """Return the ~width-char slice of a long page with the highest query-keyword
    density — so the answer region of a long article (a tail History table, a deep
    paragraph) is returned instead of just the page head. Falls back to the head
    for short text or a query with no ASCII tokens."""
    text = text or ""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    q_words = {w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    if not q_words:
        return text[:width]
    step = max(500, width // 4)
    best_start, best_score = 0, -1
    last_start = len(text) - width
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        chunk = text[start:start + width].lower()
        score = sum(1 for w in q_words if w in chunk)
        if score > best_score:
            best_score, best_start = score, start
    # Prefer the head on ties — intros carry key facts and read cleaner.
    if sum(1 for w in q_words if w in text[:width].lower()) >= best_score:
        best_start = 0
    prefix = "" if best_start == 0 else "…"
    return prefix + text[best_start + len(prefix):best_start + width]


def best_snippet(query: str, text: str, max_len: int = 240) -> str:
    """Pick the sentence from `text` with the most query-word overlap — the
    supporting evidence for a citation. Falls back to the head of the text for an
    empty query or one with no ASCII word tokens (e.g. CJK)."""
    text = (text or "").strip()
    if not text:
        return ""
    q_words = {w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    if not q_words:
        return text[:max_len]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best, best_score = "", -1
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        s_words = set(re.findall(r"[a-z0-9]{3,}", s.lower()))
        score = len(q_words & s_words)
        if score > best_score:
            best, best_score = s, score
    if best_score <= 0:
        return text[:max_len]
    return best[:max_len]
