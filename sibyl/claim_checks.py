"""Deterministic consistency checks between a fact and its cited text.

Numbers, years and dates in a fact should appear in its verified quotes, names
should appear in a cited passage, and two facts about the same subject should
not state different numbers. These catch transcription slips, misattributed
entities and silent contradictions. They compare strings, not meaning: a claim
that passes every check can still misstate its source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
# Only Latin letters, digits and underscores glue a number into a token (v3,
# SQLite3); CJK text writes numbers directly against characters (2024年).
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:[.,]\d+)*")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_WORD_RE = re.compile(r"[^\W\d_][\w'’-]*")
_SENTENCE_BREAKS = ".!?:;"
CONFLICT_OVERLAP = 0.6
_TERM_RE = re.compile(
    r"\bv?\d+(?:\.\d+)+(?:[a-z]+\d*)?(?:[-+][a-z0-9]+(?:[.-][a-z0-9]+)*)?(?!\w)"
    r"|[^\W_]+(?:['’][^\W_]+)?",
    re.UNICODE | re.IGNORECASE,
)
_STOP_WORDS = frozenset(
    "a an and are as at be been by did do does for from had has have how in is it "
    "of on or that the this to was were what when where which who why will with".split()
)


@lru_cache(maxsize=1024)
def numbers(text: str) -> frozenset[str]:
    """Numbers in the text, with thousands separators removed."""
    return frozenset(
        value.replace(",", "") if _THOUSANDS_RE.fullmatch(value) else value
        for value in _NUMBER_RE.findall(text.translate(_FULLWIDTH_DIGITS))
    )


def names(text: str) -> list[str]:
    """Capitalized words that are not merely sentence-initial, such as SQLite."""
    found = []
    for match in _WORD_RE.finditer(text):
        word = re.sub(r"['’]s$", "", match.group()).strip("-'’")
        if len(word) < 2 or not word[0].isupper():
            continue
        before = text[: match.start()].rstrip()
        initial = not before or before[-1] in _SENTENCE_BREAKS
        if initial and not any(c.isupper() or c.isdigit() for c in word[1:]):
            continue
        found.append(word)
    return list(dict.fromkeys(found))


def _comparable(text: str) -> str:
    return " " + " ".join(text.casefold().replace("’", "'").replace("-", " ").split())


def unmatched_names(text: str, cited: Iterable[str]) -> list[str]:
    """Names in the claim that no cited passage or source title mentions."""
    haystack = _comparable(" ".join(cited))
    return [
        name
        for name in names(text)
        if not re.search(
            rf"(?<!\w){re.escape(_comparable(name).strip())}(?!\w)", haystack
        )
    ]


def unquoted_numbers(text: str, quotes: Iterable[str]) -> list[str]:
    quoted = frozenset().union(*(numbers(quote) for quote in quotes))
    return sorted(numbers(text) - quoted)


@lru_cache(maxsize=1024)
def _topic(text: str) -> frozenset[str]:
    terms = (term.casefold().replace("’", "'") for term in _TERM_RE.findall(text))
    return frozenset(
        term
        for term in terms
        if term not in _STOP_WORDS and not any(c.isdigit() for c in term)
    )


def possible_conflict(first: str, second: str) -> bool:
    """Two statements about the same subject whose numbers differ."""
    a, b = numbers(first), numbers(second)
    if not a or not b or a <= b or b <= a:
        return False
    topic_a, topic_b = _topic(first), _topic(second)
    if min(len(topic_a), len(topic_b)) < 3:
        return False
    return len(topic_a & topic_b) / len(topic_a | topic_b) >= CONFLICT_OVERLAP
