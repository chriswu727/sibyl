"""Dependency-free lexical relevance scoring for keyless retrieval."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with",
}


@dataclass(frozen=True)
class LexicalCoverage:
    query_terms: int
    matched_terms: int
    score: float


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").casefold()


def _tokens(value: str) -> Set[str]:
    normalized = _normalize(value)
    tokens = {
        token for token in _WORD_RE.findall(normalized)
        if len(token) >= 2 and token not in _STOP_WORDS
    }
    for segment in _CJK_RE.findall(normalized):
        tokens.discard(segment)
        if len(segment) == 1:
            tokens.add(segment)
        else:
            tokens.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return tokens


def _phrase_text(value: str) -> str:
    normalized = _normalize(value)
    return " ".join(_WORD_RE.findall(normalized))


def lexical_relevance_scores(
    query: str,
    documents: Sequence[Tuple[str, str]],
) -> List[float]:
    """Score ``(title, text)`` documents from 0 to 1 in their original order.

    Scores combine IDF-weighted query-token coverage, title coverage, and a
    small exact-phrase signal. They measure lexical retrieval relevance, not
    probability, factual correctness, or source quality.
    """
    if not documents:
        return []

    query_tokens = _tokens(query)
    if not query_tokens:
        return [0.0] * len(documents)

    tokenized = []
    document_token_sets = []
    for title, text in documents:
        title_tokens = _tokens(title)
        body_tokens = _tokens(text)
        tokenized.append((title_tokens, body_tokens))
        document_token_sets.append(title_tokens | body_tokens)

    document_count = len(documents)
    weights = {}
    for token in query_tokens:
        frequency = sum(token in tokens for tokens in document_token_sets)
        weights[token] = math.log((document_count + 1) / (frequency + 1)) + 1
    total_weight = sum(weights.values())
    query_phrase = _phrase_text(query)

    scores = []
    for (title, text), (title_tokens, body_tokens) in zip(documents, tokenized):
        body_coverage = sum(
            weight for token, weight in weights.items() if token in body_tokens
        ) / total_weight
        title_coverage = sum(
            weight for token, weight in weights.items() if token in title_tokens
        ) / total_weight
        coverage = (
            0.5 * body_coverage
            + 0.4 * title_coverage
            + 0.1 * max(body_coverage, title_coverage)
        )
        combined_phrase_text = _phrase_text(f"{title} {text}")
        phrase_match = bool(query_phrase and query_phrase in combined_phrase_text)
        score = 0.9 * coverage + (0.1 if phrase_match else 0.0)
        scores.append(round(min(1.0, max(0.0, score)), 6))
    return scores


def lexical_query_coverage(query: str, evidence_texts: Sequence[str]) -> LexicalCoverage:
    query_tokens = _tokens(query)
    if not query_tokens:
        return LexicalCoverage(0, 0, 0.0)

    evidence_tokens = set()
    for text in evidence_texts:
        evidence_tokens.update(_tokens(text))
    matched_terms = len(query_tokens & evidence_tokens)
    return LexicalCoverage(
        query_terms=len(query_tokens),
        matched_terms=matched_terms,
        score=round(matched_terms / len(query_tokens), 6),
    )
