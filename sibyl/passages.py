"""Deterministic passage splitting with source-text offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .ranking import lexical_query_terms, lexical_relevance_scores

_SENTENCE_END = re.compile(r"[^\W\d_][.!?。！？](?:\s|$)")
_PARTIAL_WORD = re.compile(r"(?<=[A-Za-z0-9_])[A-Za-z0-9_]+")


@dataclass(frozen=True)
class TextPassage:
    text: str
    start_char: int
    end_char: int


def split_passages(
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 200,
) -> List[TextPassage]:
    if not text or max_chars <= 0:
        return []

    overlap = max(0, min(overlap_chars, max_chars // 3))
    passages = []
    start = 0
    text_length = len(text)
    while start < text_length:
        hard_end = min(text_length, start + max_chars)
        end = hard_end
        if hard_end < text_length:
            window = text[start:hard_end]
            minimum_boundary = int(len(window) * 0.6)
            boundary_ends = []
            for separator in ("\n\n", "\n", ". ", "。", "！", "？", "! ", "? "):
                position = window.rfind(separator, minimum_boundary)
                if position >= 0:
                    boundary_ends.append(start + position + len(separator))
            if boundary_ends:
                end = max(boundary_ends)

        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        passage_start = start + leading
        passage_end = end - trailing
        passage_text = text[passage_start:passage_end]
        if passage_text:
            passages.append(TextPassage(passage_text, passage_start, passage_end))

        if end >= text_length:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return passages


def passage_relevance_scores(query: str, passages: List[TextPassage]) -> List[float]:
    """Prefer query matches in prose/table rows over navigation-only matches."""
    documents = [("", passage.text) for passage in passages]
    body_documents = [
        (
            "",
            "\n".join(
                line
                for line in passage.text.splitlines()
                if len(line.strip()) >= 120
                or _SENTENCE_END.search(line)
                or line.count("|") >= 2
            ),
        )
        for passage in passages
    ]
    full_scores = lexical_relevance_scores(query, documents)
    body_scores = lexical_relevance_scores(query, body_documents)
    return [0.2 * full + 0.8 * body for full, body in zip(full_scores, body_scores)]


def select_passages(
    query: str,
    text: str,
    max_chars: int = 7000,
    max_passages: int = 3,
) -> List[TextPassage]:
    """Select bounded, disjoint excerpts with offsets into the unmodified text."""
    if not text or max_chars <= 0 or max_passages <= 0:
        return []
    count = min(max_passages, max_chars)
    passage_size = max(1, min(2500, max_chars // count))
    chunks = split_passages(text, max_chars=passage_size, overlap_chars=0)
    if len(text) <= max_chars:
        if len(chunks) > count:
            start = chunks[count - 1].start_char
            end = chunks[-1].end_char
            chunks = chunks[: count - 1] + [TextPassage(text[start:end], start, end)]
        return chunks
    if not lexical_query_terms(query):
        return split_passages(text[:max_chars], max_chars=max_chars, overlap_chars=0)

    candidates = split_passages(
        text, max_chars=passage_size, overlap_chars=min(300, passage_size // 4)
    )
    chunks = []
    for chunk in candidates:
        partial = _PARTIAL_WORD.match(text, chunk.start_char)
        start = partial.end() if partial else chunk.start_char
        start += len(text[start : chunk.end_char]) - len(
            text[start : chunk.end_char].lstrip()
        )
        if start < chunk.end_char:
            chunks.append(
                TextPassage(text[start : chunk.end_char], start, chunk.end_char)
            )
    scores = passage_relevance_scores(query, chunks)
    query_terms = lexical_query_terms(query)
    terms = [lexical_query_terms(chunk.text) for chunk in chunks]
    remaining = set(range(len(chunks)))
    selected: List[TextPassage] = []
    covered: set[str] = set()
    seen: set[str] = set()
    while remaining and len(selected) < count:
        index = max(
            remaining,
            key=lambda i: (
                scores[i]
                + 0.15 * len((terms[i] & query_terms) - covered) / len(query_terms),
                -i,
            ),
        )
        remaining.remove(index)
        chunk = chunks[index]
        if chunk.text in seen or any(
            chunk.start_char < passage.end_char and passage.start_char < chunk.end_char
            for passage in selected
        ):
            continue
        selected.append(chunk)
        seen.add(chunk.text)
        covered.update(terms[index] & query_terms)
    return sorted(selected, key=lambda passage: passage.start_char)
