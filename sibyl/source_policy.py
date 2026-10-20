"""Transparent source-routing hints, not factual or peer-review verdicts."""

from __future__ import annotations

import re
from urllib.parse import urlsplit


def normalize_domains(domains: list[str]) -> list[str]:
    if len(domains) > 16:
        raise ValueError("Use at most sixteen preferred primary domains.")
    normalized = []
    for domain in domains:
        domain = domain.lower().strip().rstrip(".")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", domain):
            raise ValueError("Preferred domains must be hostnames, not URLs.")
        if len(domain) > 253 or "." not in domain or any(not label for label in domain.split(".")):
            raise ValueError("Invalid preferred domain.")
        normalized.append(domain)
    return list(dict.fromkeys(normalized))


def matches_domain(host: str, domains: list[str] | tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def source_kind(source, preferred_domains=()) -> str:
    parsed = urlsplit(source.url)
    host, path = (parsed.hostname or "").lower(), parsed.path.lower()
    if source.content_origin == "crossref_api":
        return "scholarly_metadata"
    if matches_domain(host, ("wikipedia.org", "britannica.com")):
        return "encyclopedia"
    if matches_domain(host, ("openreview.net",)):
        return "research_candidate"
    if (matches_domain(host, ("reddit.com", "news.ycombinator.com"))
            or re.search(r"/(discussions|issues|forum)(/|$)", path)):
        return "community"
    if matches_domain(host, ("arxiv.org", "biorxiv.org", "medrxiv.org")):
        return "preprint"
    if matches_domain(host, ("aclanthology.org", "proceedings.mlr.press", "openreview.net",
                             "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "doi.org")):
        return "research_candidate"
    if matches_domain(host, preferred_domains):
        return "preferred_primary"
    if host.startswith(("docs.", "developers.", "api-docs.")) or "/docs/" in path:
        return "documentation_candidate"
    return "web"


def source_priority(source, policy="primary", preferred_domains=()) -> float:
    kind = source_kind(source, preferred_domains)
    weights = {
        "preferred_primary": 3.0, "documentation_candidate": 2.5,
        "research_candidate": 3.0, "preprint": 2.8, "web": 1.0,
        "scholarly_metadata": 0.4, "encyclopedia": -2.0, "community": -1.0,
    }
    if policy == "academic" and kind in {"research_candidate", "preprint"}:
        weights[kind] += 1.0
    relevance = max(0.0, min(1.0, source.relevance_score or 0.0))
    return weights[kind] + relevance
