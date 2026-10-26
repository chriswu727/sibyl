"""Deterministic search-query variants."""

from __future__ import annotations

import re
from typing import List
from urllib.parse import urlsplit

_TERM_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SITE_RE = re.compile(r"""(?<![-\w])site:["']?([^\s"'()\[\]{},;]+)""", re.IGNORECASE)
_QUOTED_RE = re.compile(r"""["“]([^"”]+)["”]|(?<!\w)'([^']{3,})'(?!\w)""")
_SEARCH_OPERATOR_RE = re.compile(
    r"(?<!\w)-?(?:site|filetype|intitle|inurl):|(?:^|\s)-\w", re.IGNORECASE
)
_QUESTION_WORDS = {"how", "what", "when", "where", "which", "who", "why"}
_RELATIVE_CHAIN_RE = re.compile(
    r"\b(?:company|country|institution|organization|person)\s+"
    r"(?:that|whose)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_QUESTION_RE = re.compile(
    r"[.!?]\s+(?:how|what|when|where|which|who|why)\b",
    re.IGNORECASE,
)
_HISTORICAL_ROLE_RE = re.compile(
    r"^\s*who\s+(?:is|was|served\s+as)\s+(?:the\s+)?"
    r"(?P<role>[^?]{1,60}?)\s+(?:of|at|for)\b.*?"
    r"\bin\s+(?P<year>[12]\d{3})\s*\??$",
    re.IGNORECASE,
)
_SEARCH_STOP_WORDS = _QUESTION_WORDS | {
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


def query_site_hosts(query: str) -> list[str]:
    hosts = []
    for value in _SITE_RE.findall(query):
        try:
            host = urlsplit(value if "://" in value else "https://" + value).hostname
        except ValueError:
            continue
        if host:
            hosts.append(host.rstrip(".").lower())
    return list(dict.fromkeys(hosts))


def focused_site_query(query: str, host: str) -> str:
    focused = _SITE_RE.sub(" ", query)
    focused = re.sub(r"\b(?:OR|AND)\b|[()\[\]{}\"']", " ", focused)
    return f"{' '.join(focused.split())} site:{host}".strip()


def search_query_variants(query: str) -> List[str]:
    clean = " ".join(str(query or "").split())
    if not clean:
        return []
    if _SEARCH_OPERATOR_RE.search(clean):
        return [clean]

    terms = _TERM_RE.findall(clean)
    if not terms or not any(term.casefold() in _QUESTION_WORDS for term in terms):
        return [clean]

    quoted_match = _QUOTED_RE.search(clean)
    if quoted_match:
        quoted = " ".join((quoted_match.group(1) or quoted_match.group(2)).split())
        if len(_TERM_RE.findall(quoted)) >= 3:
            return [clean, quoted]

    focused_terms = [
        term for term in terms if term.casefold() not in _SEARCH_STOP_WORDS
    ]
    focused = " ".join(focused_terms)
    if len(focused_terms) < 3 or focused.casefold() == clean.casefold():
        return [clean]
    return [clean, focused]


def query_requires_decomposition(query: str) -> bool:
    clean = " ".join(str(query or "").split())
    if not clean:
        return False
    return bool(
        _RELATIVE_CHAIN_RE.search(clean) or _CONTEXTUAL_QUESTION_RE.search(clean)
    )


def historical_role_requirement(query: str) -> tuple[str, int] | None:
    match = _HISTORICAL_ROLE_RE.match(" ".join(str(query or "").split()))
    if not match:
        return None
    role = " ".join(match.group("role").split()).casefold()
    if not role or len(_TERM_RE.findall(role)) > 4:
        return None
    return role, int(match.group("year"))
