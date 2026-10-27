"""Conservative period-matching hints for retrieval, not freshness verdicts."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_CURRENT_RE = re.compile(
    r"\b(?:latest|current|currently|today|now)\b|当前|目前|最新|现在", re.IGNORECASE
)
_FISCAL_RE = re.compile(r"\bfiscal\b|\bfy\s*\d", re.IGNORECASE)


def period_match(
    query: str, title: str, url: str, published_at: str | None = None
) -> int:
    years = set(_YEAR_RE.findall(query))
    if not years and _CURRENT_RE.search(query):
        years = {str(datetime.now(timezone.utc).year)}
    if not years:
        return 0
    title_years = set(_YEAR_RE.findall(title))
    published_year = (published_at or "")[:4]
    if _FISCAL_RE.search(title) and not _FISCAL_RE.search(query):
        return (1 if published_year in years else -1) if published_year.isdigit() else 0
    if title_years:
        if title_years & years:
            return 1
        return -1
    if published_year.isdigit():
        return 1 if published_year in years else -1
    path_years = set(_YEAR_RE.findall(urlsplit(url).path))
    return (1 if path_years & years else -1) if path_years else 0
