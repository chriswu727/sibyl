"""Host-authored claims, evidence integrity and explicit reasoning dependencies."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .research_errors import ResearchRequestError

if TYPE_CHECKING:
    from .research_state import ResearchNotebook


class RecommendationCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    preference: str = Field(min_length=1, max_length=300)
    origin: Literal["user", "host_assumption"]


class ResearchClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str = Field(pattern=r"^Q[0-5]\.C[1-9][0-9]{0,2}$")
    kind: Literal["fact", "inference", "recommendation"]
    status: Literal["supported", "disputed", "unresolved"]
    text: str = Field(min_length=1, max_length=1000)
    supporting_passage_ids: list[str] = Field(default_factory=list, max_length=12)
    challenging_passage_ids: list[str] = Field(default_factory=list, max_length=12)
    basis_claim_ids: list[str] = Field(default_factory=list, max_length=12)
    rationale: str = Field(default="", max_length=1000)
    assumptions: list[str] = Field(default_factory=list, max_length=4)
    criteria: list[RecommendationCriterion] = Field(default_factory=list, max_length=6)


def claim_index(notebook: ResearchNotebook) -> dict[str, tuple[str, dict[str, Any]]]:
    return {
        claim["claim_id"]: (question_id, claim)
        for question_id, review in notebook.reviews.items()
        for claim in review.get("claims", [])
    }


def claim_fingerprint(claim: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(claim, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def dependency_ids(
    claim_id: str, index: dict[str, tuple[str, dict[str, Any]]]
) -> set[str]:
    found: set[str] = set()

    def visit(current: str) -> None:
        for dependency in index[current][1]["basis_claim_ids"]:
            if dependency not in found:
                found.add(dependency)
                visit(dependency)

    visit(claim_id)
    return found


def validate_claims(
    notebook: ResearchNotebook,
    question_id: str,
    values: list[dict[str, Any]],
    evidence_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if len(values) > 12:
        raise ResearchRequestError(
            "Record at most twelve material claims per question."
        )
    claims = []
    for value in values:
        try:
            claim = ResearchClaim.model_validate(value).model_dump()
        except ValidationError:
            raise ResearchRequestError(
                "Invalid claim record; follow the ResearchClaim schema.",
                code="invalid_claim",
                field="claims",
            ) from None
        if not claim["claim_id"].startswith(question_id + "."):
            raise ResearchRequestError(
                "Claim IDs must start with their question ID, for example Q1.C1."
            )
        for key in (
            "supporting_passage_ids",
            "challenging_passage_ids",
            "basis_claim_ids",
        ):
            if any(not isinstance(ref, str) or not ref.strip() for ref in claim[key]):
                raise ResearchRequestError("Claim references must be nonempty IDs.")
            claim[key] = list(dict.fromkeys(claim[key]))
        support, challenge = (
            claim["supporting_passage_ids"],
            claim["challenging_passage_ids"],
        )
        if set(support) & set(challenge):
            raise ResearchRequestError(
                "One passage cannot both support and challenge the same claim."
            )
        if not set(support + challenge) <= evidence_ids:
            raise ResearchRequestError(
                "Claim references an unknown or snippet-only passage.",
                code="invalid_claim",
                claim_id=claim["claim_id"],
                field="claims.supporting_passage_ids"
                if not set(support) <= evidence_ids
                else "claims.challenging_passage_ids",
            )
        if claim["status"] == "supported" and challenge:
            raise ResearchRequestError(
                "A claim with unresolved challenging evidence must be disputed or unresolved."
            )
        if claim["status"] == "unresolved" and not claim["rationale"]:
            raise ResearchRequestError(
                "Unresolved claims must explain what evidence is missing or disputed.",
                code="invalid_claim",
                claim_id=claim["claim_id"],
                field="claims.rationale",
            )
        if claim["status"] == "disputed" and (
            not challenge or not (support or claim["basis_claim_ids"])
        ):
            raise ResearchRequestError(
                "Disputed claims require both a supporting basis and challenging evidence.",
                code="invalid_claim",
                claim_id=claim["claim_id"],
                field="claims.challenging_passage_ids"
                if not challenge
                else "claims.supporting_passage_ids"
                if claim["kind"] == "fact"
                else "claims.basis_claim_ids",
            )
        if claim["kind"] == "fact":
            if claim["basis_claim_ids"] or claim["criteria"]:
                raise ResearchRequestError(
                    "Facts cite original passages; reasoning dependencies and criteria belong to inferences/recommendations."
                )
            if claim["status"] == "supported" and not support:
                raise ResearchRequestError(
                    "Supported facts require exact evidence passage IDs.",
                    code="invalid_claim",
                    claim_id=claim["claim_id"],
                    field="claims.supporting_passage_ids",
                )
        else:
            if not claim["rationale"]:
                raise ResearchRequestError(
                    "Inferences and recommendations must explain their reasoning.",
                    code="invalid_claim",
                    claim_id=claim["claim_id"],
                    field="claims.rationale",
                )
            if claim["status"] != "unresolved" and not claim["basis_claim_ids"]:
                raise ResearchRequestError(
                    "Inferences and recommendations must identify their factual/reasoning basis.",
                    code="invalid_claim",
                    claim_id=claim["claim_id"],
                    field="claims.basis_claim_ids",
                )
        if claim["kind"] == "recommendation" and not claim["criteria"]:
            raise ResearchRequestError(
                "Recommendations must state criteria and whether they come from the user or a host assumption.",
                code="invalid_claim",
                claim_id=claim["claim_id"],
                field="claims.criteria",
            )
        if claim["kind"] != "recommendation" and claim["criteria"]:
            raise ResearchRequestError("Only recommendations have selection criteria.")
        if any(
            not isinstance(item, str) or not item.strip() or len(item) > 300
            for item in claim["assumptions"]
        ):
            raise ResearchRequestError(
                "Assumptions must be nonempty strings of at most 300 characters."
            )
        claim["assumptions"] = [item.strip() for item in claim["assumptions"]]
        authored_text = [
            claim["text"],
            claim["rationale"],
            *claim["assumptions"],
            *(
                value
                for criterion in claim["criteria"]
                for value in (criterion["name"], criterion["preference"])
            ),
        ]
        if any(
            re.search(r"\[Source\s+\d+\]", text, re.IGNORECASE)
            for text in authored_text
        ):
            raise ResearchRequestError(
                "Use passage IDs for evidence links; report source numbers are generated at completion."
            )
        claims.append(claim)
    ids = [claim["claim_id"] for claim in claims]
    if len(set(ids)) != len(ids):
        raise ResearchRequestError("Claim IDs must be unique within an assessment.")
    index = {
        key: item
        for key, item in claim_index(notebook).items()
        if item[0] != question_id
    }
    index.update({claim["claim_id"]: (question_id, claim) for claim in claims})
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(claim_id: str) -> None:
        if claim_id not in index:
            raise ResearchRequestError(
                "Claim basis references an unknown claim; preserve claims referenced by other questions."
            )
        if claim_id in visiting:
            raise ResearchRequestError(
                "Claim reasoning dependencies must not contain cycles."
            )
        if claim_id in visited:
            return
        visiting.add(claim_id)
        for basis in index[claim_id][1]["basis_claim_ids"]:
            visit(basis)
        visiting.remove(claim_id)
        visited.add(claim_id)

    for claim_id in index:
        visit(claim_id)
    for claim in claims:
        if claim["status"] == "supported" and any(
            index[key][1]["status"] != "supported"
            for key in dependency_ids(claim["claim_id"], index)
        ):
            raise ResearchRequestError(
                "A supported inference or recommendation cannot depend on unresolved or disputed claims."
            )
    dependencies = set().union(*(dependency_ids(claim_id, index) for claim_id in ids))
    return claims, {
        key: claim_fingerprint(index[key][1]) for key in sorted(dependencies)
    }


def review_is_stale(notebook: ResearchNotebook, question_id: str) -> bool:
    from .research_review import question_attempts, questions

    review = notebook.reviews[question_id]
    attempts = question_attempts(
        notebook, question_id, questions(notebook)[question_id]
    )
    return any(
        position >= review["reviewed_attempts"] and attempt in attempts
        for position, attempt in enumerate(notebook.attempts)
    )


def claim_gaps(notebook: ResearchNotebook, question_id: str) -> list[str]:
    review = notebook.reviews[question_id]
    claims = review.get("claims", [])
    if not claims:
        return (
            [
                "Record material facts, inferences and recommendations with claim-level evidence links."
            ]
            if notebook.claim_contract_version >= 2
            else []
        )
    index = claim_index(notebook)
    gaps = []
    for claim in claims:
        claim_id = claim["claim_id"]
        if claim["status"] != "supported":
            gaps.append(f"{claim_id}: the host marked this claim {claim['status']}.")
        for basis in sorted(dependency_ids(claim_id, index)):
            owner, premise = index[basis]
            if premise["status"] != "supported":
                gaps.append(f"{claim_id}: basis {basis} is {premise['status']}.")
            if review_is_stale(notebook, owner):
                gaps.append(
                    f"{claim_id}: basis {basis} needs reassessment after new retrieval."
                )
    for basis, fingerprint in review.get("basis_fingerprints", {}).items():
        if basis not in index or claim_fingerprint(index[basis][1]) != fingerprint:
            gaps.append(f"Basis {basis} changed; reassess dependent claims.")
    return list(dict.fromkeys(gaps))


def assessment_current(notebook: ResearchNotebook, claim_id: str) -> bool:
    index = claim_index(notebook)
    owner = index[claim_id][0]
    if review_is_stale(notebook, owner):
        return False
    dependencies = dependency_ids(claim_id, index)
    if any(review_is_stale(notebook, index[key][0]) for key in dependencies):
        return False
    recorded = notebook.reviews[owner].get("basis_fingerprints", {})
    return all(
        recorded.get(key) == claim_fingerprint(index[key][1]) for key in dependencies
    )


def render_claims(
    notebook: ResearchNotebook, selected_ids: list[str] | None
) -> tuple[list[dict[str, Any]], list[str], str]:
    index = claim_index(notebook)
    selected = list(dict.fromkeys(selected_ids if selected_ids is not None else index))
    if not selected or len(selected) > 32 or not set(selected) <= index.keys():
        raise ResearchRequestError(
            "Select 1-32 existing claim IDs from the research reviews."
        )
    included = set(selected)
    included.update(
        key for key, (_, claim) in index.items() if claim["status"] != "supported"
    )
    for key in list(included):
        included.update(dependency_ids(key, index))
    citation_map = {
        p.citation_id: f"[Source {number}]"
        for number, source in enumerate(notebook.selected_sources(), 1)
        for p in source.evidence
    }
    records, findings, details = [], [], []
    for key, (owner, claim) in index.items():
        if key not in included:
            continue
        record = {
            **claim,
            "question_id": owner,
            "review_current": assessment_current(notebook, key),
        }
        records.append(record)
        references = set(
            claim["supporting_passage_ids"] + claim["challenging_passage_ids"]
        )
        for basis in dependency_ids(key, index):
            premise = index[basis][1]
            references.update(
                premise["supporting_passage_ids"] + premise["challenging_passage_ids"]
            )
        if not references <= citation_map.keys():
            raise ResearchRequestError(
                "Claim evidence is outside the selected source window; reassess the evidence links."
            )
        citations = " ".join(
            dict.fromkeys(citation_map[ref] for ref in sorted(references))
        )
        status_label = (
            claim["status"]
            if record["review_current"]
            else claim["status"] + ", needs reassessment"
        )
        line = f"{key} — {claim['kind']} ({status_label}): {claim['text']}"
        if citations:
            line += " " + citations
        if key in selected:
            findings.append(line)
        details.append(line)
        if claim["basis_claim_ids"]:
            details.append("Basis: " + ", ".join(claim["basis_claim_ids"]))
        if claim["rationale"]:
            details.append("Reasoning: " + claim["rationale"])
        details.extend("Assumption: " + item for item in claim["assumptions"])
        details.extend(
            f"Criterion ({item['origin']}): {item['name']} — {item['preference']}"
            for item in claim["criteria"]
        )
        if not record["review_current"]:
            details.append("This assessment needs review after new retrieval.")
        details.append("")
    return records, findings, "\n".join(details).strip()
