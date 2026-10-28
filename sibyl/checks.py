"""Check a ledger's quotes against the pages they cite.

Each cited URL is read once over the safe network boundary. A quote counts as
found when its text, after Unicode, case, quotation-mark, dash and whitespace
normalization, occurs in the page text Sibyl extracted; an ellipsis splits a
quote into parts that must occur in order. The checks compare strings, not
meaning: a found quote can still be taken out of context, and a claim that
passes every check can still misstate its source.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional
from urllib.parse import urlsplit

from . import __version__
from .claim_checks import possible_conflict, unmatched_names, unquoted_numbers
from .ledger import Claim, Ledger
from .scraper import WebPage, scrape_urls

CHECK_FORMAT = "sibyl-check/1"
NAME_WINDOW = 1000
MIN_QUOTE_CHARS = 20
MAX_CONFLICTS = 5

# Blocking issues stop a report; warnings are disclosed in it.
BLOCKING = {
    "quote_not_found": "a quote does not occur in the text read from its page",
    "source_unreadable": "a cited page could not be read, so its quote is unverified",
    "number_not_quoted": "the claim states a number that none of its supporting quotes contains",
    "basis_has_issues": "an inference rests on a claim with blocking issues",
}
WARNINGS = {
    "single_source": "all supporting quotes come from one site",
    "name_not_in_sources": "a name in the claim does not appear near its quotes",
    "possible_conflict": "another claim about the same subject states different numbers",
    "short_quote": "a quote is too short to identify a statement",
}

_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "«": '"',
        "»": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "­": None,
        "​": None,
        "‌": None,
        "‍": None,
        "⁠": None,
        "﻿": None,
    }
)
_ELLIPSIS_RE = re.compile(r"\s*(?:\[\s*(?:\.\s*){3}\]|\[…\]|\.{3}|…)\s*")


def normalize(text: str) -> str:
    """The comparable form of page text and quotes."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).casefold()
    return " ".join(text.split())


def locate(quote: str, page: str) -> Optional[int]:
    """Offset of the quote in the normalized page text, or None when absent."""
    parts = [normalize(part) for part in _ELLIPSIS_RE.split(quote)]
    parts = [part for part in parts if part]
    if not parts:
        return None
    start = position = page.find(parts[0])
    if start < 0:
        return None
    position += len(parts[0])
    for part in parts[1:]:
        found = page.find(part, position)
        if found < 0:
            return None
        position = found + len(part)
    return start


def site(url: str) -> str:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        # The reader already reports a malformed URL as unreadable.
        return url
    return host.lower().removeprefix("www.")


@dataclass
class SourceRead:
    url: str
    title: str = ""
    source_type: str = "web"
    text_sha256: str = ""
    chars: int = 0
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    normalized: str = field(default="", repr=False)

    @classmethod
    def from_page(cls, page: WebPage) -> "SourceRead":
        text = page.text if not page.error else ""
        return cls(
            url=page.url,
            title=page.title,
            source_type=page.source_type,
            text_sha256=hashlib.sha256(text.encode()).hexdigest() if text else "",
            chars=len(text),
            error=page.error or ("" if text else "no readable text"),
            warnings=list(page.extraction_warnings),
            normalized=normalize(text),
        )

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "source_type": self.source_type,
            "text_sha256": self.text_sha256,
            "chars": self.chars,
            "error": self.error,
            "warnings": self.warnings,
        }


Reader = Callable[[list[str]], Awaitable[list[WebPage]]]


async def read_sources(
    urls: list[str], reader: Optional[Reader] = None
) -> dict[str, tuple[SourceRead, str]]:
    """Each URL's read summary and its raw text, keyed by the cited URL."""
    pages = await (reader or scrape_urls)(urls)
    result = {}
    for url, page in zip(urls, pages):
        read = SourceRead.from_page(page)
        read.url = url
        result[url] = (read, "" if page.error else page.text)
    return result


def _evidence_result(claim: Claim, sources: dict) -> tuple[list[dict], set[str]]:
    issues: set[str] = set()
    results = []
    for item in claim.evidence:
        read, _ = sources[item.url]
        offset = None if read.error else locate(item.quote, read.normalized)
        if read.error:
            issues.add("source_unreadable")
        elif offset is None:
            issues.add("quote_not_found")
        if len(normalize(item.quote)) < MIN_QUOTE_CHARS:
            issues.add("short_quote")
        results.append(
            {
                "url": item.url,
                "role": item.role,
                "found": offset is not None,
                "offset": offset,
                "text_sha256": read.text_sha256,
            }
        )
    return results, issues


def _near_quotes(claim: Claim, sources: dict) -> list[str]:
    context = []
    for item in claim.evidence:
        if item.role != "supports":
            continue
        read, _ = sources[item.url]
        context.append(item.quote)
        context.append(read.title)
        offset = locate(item.quote, read.normalized) if not read.error else None
        if offset is not None:
            context.append(
                read.normalized[max(0, offset - NAME_WINDOW) : offset + NAME_WINDOW]
            )
    return context


def check(ledger: Ledger, ledger_sha256: str, sources: dict) -> dict:
    """The check result for a ledger, given every cited URL's read."""
    by_id = {claim.id: claim for claim in ledger.claims}
    claims: dict[str, dict] = {}
    for claim in ledger.claims:
        evidence, issues = _evidence_result(claim, sources)
        details: dict = {}
        supporting = [item for item in claim.evidence if item.role == "supports"]
        if claim.kind == "fact" and supporting:
            numbers = unquoted_numbers(claim.text, [item.quote for item in supporting])
            if numbers:
                issues.add("number_not_quoted")
                details["unquoted_numbers"] = numbers
            names = unmatched_names(claim.text, _near_quotes(claim, sources))
            if names:
                issues.add("name_not_in_sources")
                details["unmatched_names"] = names
            if len({site(item.url) for item in supporting}) == 1:
                issues.add("single_source")
        claims[claim.id] = {
            "id": claim.id,
            "evidence": evidence,
            "issues": issues,
            "details": details,
        }

    stated = [
        claim
        for claim in ledger.claims
        if claim.kind == "fact" and claim.status != "unresolved"
    ]
    for claim in stated:
        conflicts = [
            other.id
            for other in stated
            if other.id != claim.id and possible_conflict(claim.text, other.text)
        ][:MAX_CONFLICTS]
        if conflicts:
            claims[claim.id]["issues"].add("possible_conflict")
            claims[claim.id]["details"]["conflicts_with"] = conflicts

    # The ledger rejects cycles, so each claim's answer can be cached.
    blocked_cache: dict[str, bool] = {}

    def blocked(item: str) -> bool:
        if item not in blocked_cache:
            blocked_cache[item] = bool(claims[item]["issues"] & BLOCKING.keys()) or any(
                blocked(parent) for parent in by_id[item].based_on
            )
        return blocked_cache[item]

    for claim in ledger.claims:
        if claim.kind == "inference" and any(blocked(p) for p in claim.based_on):
            claims[claim.id]["issues"].add("basis_has_issues")

    rows = []
    for claim in ledger.claims:
        row = claims[claim.id]
        issues = row.pop("issues")
        row["blocking"] = sorted(issues & BLOCKING.keys())
        row["warnings"] = sorted(issues & WARNINGS.keys())
        rows.append(row)
    notes = []
    if ledger.counterevidence is None:
        notes.append("no counterevidence search is recorded")
    return {
        "check": CHECK_FORMAT,
        "sibyl_version": __version__,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ledger_sha256": ledger_sha256,
        "passed": not any(row["blocking"] for row in rows),
        "sources": [read.to_dict() for read, _ in sources.values()],
        "claims": rows,
        "notes": notes,
    }


def cited_urls(ledger: Ledger) -> list[str]:
    return list(
        dict.fromkeys(item.url for claim in ledger.claims for item in claim.evidence)
    )


def run_check(
    ledger: Ledger,
    ledger_sha256: str,
    *,
    reader: Optional[Reader] = None,
    pages_dir: Optional[Path] = None,
) -> dict:
    """Read every cited page, check the ledger, and optionally keep page texts."""
    sources = asyncio.run(read_sources(cited_urls(ledger), reader))
    result = check(ledger, ledger_sha256, sources)
    if pages_dir is not None:
        pages_dir.mkdir(parents=True, exist_ok=True)
        for read, text in sources.values():
            if text:
                (pages_dir / f"{read.text_sha256}.txt").write_text(text, "utf-8")
    return result


def summary(result: dict) -> str:
    """A plain-text account of a check result for the host to act on."""
    lines = []
    unreadable = [s for s in result["sources"] if s["error"]]
    lines.append(
        f"Read {len(result['sources']) - len(unreadable)} of "
        f"{len(result['sources'])} cited pages."
    )
    for source in unreadable:
        lines.append(f"  unreadable: {source['url']} ({source['error']})")
    for row in result["claims"]:
        if not row["blocking"] and not row["warnings"]:
            continue
        lines.append(f"Claim {row['id']}:")
        for issue in row["blocking"]:
            lines.append(f"  BLOCKING {issue}: {BLOCKING[issue]}")
        for issue in row["warnings"]:
            lines.append(f"  warning {issue}: {WARNINGS[issue]}")
        for item in row["evidence"]:
            if not item["found"]:
                lines.append(f"    not verified: {item['url']}")
        for key, value in row["details"].items():
            lines.append(f"    {key}: {', '.join(value)}")
    for note in result["notes"]:
        lines.append(f"Note: {note}.")
    lines.append(
        "PASSED: every quote was found and no blocking issue remains."
        if result["passed"]
        else "FAILED: fix each BLOCKING claim (quote the page exactly, cite a "
        "readable page, or mark the claim unresolved with a note), then rerun."
    )
    return "\n".join(lines)
