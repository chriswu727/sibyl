"""Deterministic passage splitting with source-text offsets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


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
