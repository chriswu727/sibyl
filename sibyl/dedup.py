"""Canonical-URL near-duplicate removal for scraped pages.

Two URLs that differ only in scheme, a leading www, tracking params, fragment,
or a trailing slash point at the same document — feeding both to synthesis wastes
context and skews consensus. Pure-Python, keyless, can only ever remove sources.
"""
from __future__ import annotations

from typing import List
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from .scraper import WebPage

_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
             "utm_id", "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src",
             "igshid", "s", "spm", "_hsenc", "_hsmi"}


def canonical_url(url: str) -> str:
    """Normalize a URL to a canonical key for dedup."""
    if not url or not url.startswith("http"):
        return (url or "").strip().lower()
    try:
        parts = urlsplit(url)
        host = parts.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = parts.path.rstrip("/") or "/"
        query = urlencode(sorted(
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
            if k.lower() not in _TRACKING
        ))
        # scheme dropped from the key (http vs https = same doc)
        return urlunsplit(("", host, path, query, ""))
    except Exception:
        return url.strip().lower()


def dedup_pages(pages: List[WebPage]) -> List[WebPage]:
    """Keep one page per canonical URL: first occurrence wins its position, but
    the longest .text wins the content (a fuller scrape beats a thin duplicate)."""
    order: List[str] = []
    by_key = {}
    for p in pages:
        key = canonical_url(p.url)
        if key not in by_key:
            by_key[key] = p
            order.append(key)
        elif len(p.text or "") > len(by_key[key].text or ""):
            # keep the first occurrence's URL/title/position, take the fuller text
            kept = by_key[key]
            by_key[key] = WebPage(
                url=kept.url,
                title=kept.title,
                text=p.text,
                error=kept.error,
                content_origin=p.content_origin,
            )
    return [by_key[k] for k in order]
