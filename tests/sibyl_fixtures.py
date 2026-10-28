"""Shared ledgers and a fake page reader for offline tests."""

from __future__ import annotations

import copy

from sibyl.scraper import WebPage

PAGES = {
    "https://www.sqlite.org/wal.html": (
        "Write-Ahead Logging",
        "The default method by which SQLite implements atomic commit and "
        "rollback is a rollback journal. Beginning with version 3.7.0 "
        "(2010-07-21), a new “Write-Ahead Log” option is available. "
        "WAL provides more concurrency as readers do not block writers and a "
        "writer does not block readers.",
    ),
    "https://en.wikipedia.org/wiki/SQLite": (
        "SQLite - Wikipedia",
        "SQLite added write-ahead logging in version 3.7.0, released in July "
        "2010.\nIn WAL mode, readers and writers can proceed concurrently.",
    ),
    "https://example.com/forum": (
        "Forum thread",
        "WAL mode still allows only one writer at a time, so write-heavy "
        "workloads see little benefit.",
    ),
}


def page_reader(pages=None, failures=None):
    pages = PAGES if pages is None else pages
    failures = failures or {}

    async def read(urls):
        result = []
        for url in urls:
            if url in failures:
                result.append(WebPage(url, "", "", error=failures[url]))
            else:
                title, text = pages[url]
                result.append(WebPage(url, title, text))
        return result

    return read


LEDGER = {
    "ledger": "sibyl-ledger/1",
    "question": "When did SQLite add write-ahead logging, and what does it change?",
    "claims": [
        {
            "id": "wal-version",
            "text": "SQLite added write-ahead logging in version 3.7.0 in 2010.",
            "evidence": [
                {
                    "url": "https://www.sqlite.org/wal.html",
                    "quote": "Beginning with version 3.7.0 (2010-07-21), a new "
                    '"Write-Ahead Log" option is available.',
                },
                {
                    "url": "https://en.wikipedia.org/wiki/SQLite",
                    "quote": "SQLite added write-ahead logging in version 3.7.0, "
                    "released in July 2010.",
                },
            ],
        },
        {
            "id": "wal-concurrency",
            "text": "In WAL mode readers do not block writers.",
            "status": "disputed",
            "evidence": [
                {
                    "url": "https://www.sqlite.org/wal.html",
                    "quote": "readers do not block writers and a writer does not "
                    "block readers",
                },
                {
                    "url": "https://example.com/forum",
                    "quote": "WAL mode still allows only one writer at a time",
                    "role": "challenges",
                },
            ],
        },
        {
            "id": "advice",
            "kind": "inference",
            "text": "WAL suits read-heavy workloads on SQLite 3.7.0 or later.",
            "based_on": ["wal-version", "wal-concurrency"],
            "rationale": "Readers proceed during writes, but writers still serialize.",
        },
        {
            "id": "benchmarks",
            "text": "How much WAL speeds up a typical web application.",
            "status": "unresolved",
            "note": "No independent benchmark was found.",
        },
    ],
    "counterevidence": {
        "searched_for": "Reports that WAL does not improve concurrency.",
        "found": "A forum thread notes writers still serialize.",
    },
    "gaps": ["Performance on network file systems."],
}


def ledger_data(**changes):
    data = copy.deepcopy(LEDGER)
    data.update(changes)
    return data
