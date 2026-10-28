"""The evidence ledger: a host-written JSON record of claims and their quotes.

The ledger is the only input Sibyl needs. The host researches with its own
tools and records each claim with the URLs and verbatim quotes it relies on;
Sibyl validates the structure here and checks the quotes in ``checks``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

LEDGER_FORMAT = "sibyl-ledger/1"
MAX_LEDGER_BYTES = 1024 * 1024


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Evidence(_Strict):
    url: str = Field(min_length=1, max_length=2048)
    quote: str = Field(min_length=1, max_length=2000)
    role: Literal["supports", "challenges"] = "supports"


class Claim(_Strict):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    text: str = Field(min_length=1, max_length=2000)
    kind: Literal["fact", "inference"] = "fact"
    status: Literal["supported", "disputed", "unresolved"] = "supported"
    evidence: list[Evidence] = Field(default_factory=list, max_length=20)
    based_on: list[str] = Field(default_factory=list, max_length=20)
    rationale: str = Field(default="", max_length=2000)
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _shape(self) -> "Claim":
        roles = {item.role for item in self.evidence}
        if self.kind == "inference":
            if self.evidence:
                raise ValueError(
                    "an inference cites the claims it rests on in based_on, not evidence"
                )
            if self.status == "disputed":
                raise ValueError(
                    "an inference cannot be disputed; record the disputed fact it rests on"
                )
            if self.status != "unresolved" and not (self.based_on and self.rationale):
                raise ValueError("an inference needs based_on and a rationale")
        elif self.based_on:
            raise ValueError("based_on belongs to inferences; facts cite evidence")
        elif self.status == "supported" and "supports" not in roles:
            raise ValueError("a supported fact needs at least one supporting quote")
        elif self.status == "disputed" and roles != {"supports", "challenges"}:
            raise ValueError(
                "a disputed fact needs at least one supporting and one challenging quote"
            )
        if self.status == "unresolved" and not self.note:
            raise ValueError("an unresolved claim needs a note saying what is missing")
        return self


class Counterevidence(_Strict):
    searched_for: str = Field(min_length=1, max_length=2000)
    found: str = Field(default="", max_length=2000)


class Ledger(_Strict):
    ledger: Literal["sibyl-ledger/1"] = "sibyl-ledger/1"
    question: str = Field(min_length=1, max_length=2000)
    claims: list[Claim] = Field(min_length=1, max_length=200)
    counterevidence: Optional[Counterevidence] = None
    gaps: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _references(self) -> "Ledger":
        ids = [claim.id for claim in self.claims]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate claim ids: {', '.join(duplicates)}")
        known = set(ids)
        for claim in self.claims:
            missing = [item for item in claim.based_on if item not in known]
            if missing:
                raise ValueError(
                    f"claim {claim.id} is based on unknown claims: {', '.join(missing)}"
                )
        basis = {claim.id: claim.based_on for claim in self.claims}

        def visit(item: str, path: tuple[str, ...]) -> None:
            if item in path:
                cycle = " -> ".join(path[path.index(item) :] + (item,))
                raise ValueError(f"circular based_on: {cycle}")
            for parent in basis[item]:
                visit(parent, path + (item,))

        for item in ids:
            visit(item, ())
        return self


class LedgerError(ValueError):
    """The ledger file is missing, unreadable or does not follow the format."""


def _describe(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in item["loc"]
        ).lstrip(".")
        message = item["msg"].removeprefix("Value error, ")
        lines.append(f"{location or 'ledger'}: {message}")
    return "\n".join(lines)


def read_ledger(path: Path) -> tuple[Ledger, str]:
    """The parsed ledger and the SHA-256 of its exact bytes."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LedgerError(f"cannot read {path}: {exc.strerror or exc}") from None
    if len(data) > MAX_LEDGER_BYTES:
        raise LedgerError(f"{path} exceeds {MAX_LEDGER_BYTES} bytes")
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"{path} is not valid JSON: {exc}") from None
    try:
        ledger = Ledger.model_validate(raw)
    except ValidationError as exc:
        raise LedgerError(_describe(exc)) from None
    return ledger, hashlib.sha256(data).hexdigest()


def check_path(ledger_path: Path) -> Path:
    """Where the check result for a ledger is written: beside it."""
    return ledger_path.with_name(ledger_path.stem + ".check.json")


TEMPLATE = {
    "ledger": LEDGER_FORMAT,
    "question": "What does the research question ask?",
    "claims": [
        {
            "id": "c1",
            "text": "A factual claim, as it will appear in the answer.",
            "evidence": [
                {
                    "url": "https://example.org/primary-source",
                    "quote": "Exact text copied from that page that states the claim.",
                }
            ],
        },
        {
            "id": "c2",
            "text": "A claim that sources disagree on.",
            "status": "disputed",
            "evidence": [
                {"url": "https://example.org/a", "quote": "Text that supports it."},
                {
                    "url": "https://example.net/b",
                    "quote": "Text that contradicts it.",
                    "role": "challenges",
                },
            ],
        },
        {
            "id": "c3",
            "kind": "inference",
            "text": "A conclusion drawn from the facts above.",
            "based_on": ["c1", "c2"],
            "rationale": "Why the cited claims lead to this conclusion.",
        },
        {
            "id": "c4",
            "text": "Something the sources did not establish.",
            "status": "unresolved",
            "note": "What was searched and what is still missing.",
        },
    ],
    "counterevidence": {
        "searched_for": "The search for evidence against the main conclusion.",
        "found": "What it found, or that it found nothing.",
    },
    "gaps": ["A question the research could not answer."],
}
