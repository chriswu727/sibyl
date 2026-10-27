"""One format-neutral report document shared by the Markdown, text and PDF renderers.

The document decides which sections a report has and in what order; renderers
only decide how each block looks. JSON exports serialize the report record
itself, so every format carries the same content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .privacy import display_url
from .report_models import ResearchReport, Source
from .report_references import reference_groups
from .report_sections import (
    CLAIM_SECTION_NOTICE,
    CLAIM_SECTION_TITLE,
    LEGACY_VERDICT_NOTICE,
    claim_report_body,
    escape_quote_lines,
    legacy_verdict_tag,
)

PARTIAL_NOTICE = "Partial report: unresolved research gaps remain."
UNVERIFIED_NOTICE = "Claim verification was not performed for this report."
REVIEW_NOTICE = (
    "Host-declared assessments; passage links do not establish semantic truth."
)


@dataclass(frozen=True)
class Block:
    """A unit of report content.

    text: prose that may carry the host's Markdown; notice: a caution;
    item: a bullet; claims: the rendered claim ledger, whose quotes are
    untrusted source text; reference: a numbered source; chart: an image path.
    """

    kind: Literal["text", "notice", "item", "claims", "reference", "chart"]
    text: str = ""
    number: int = 0
    source: Source | None = None


@dataclass(frozen=True)
class Section:
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True)
class ReportDocument:
    title: str
    completed: bool
    generated: str = ""
    notice: str = ""
    sections: list[Section] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

    def texts(self) -> list[str]:
        """Every string a renderer must be able to show."""
        return [
            self.title,
            self.notice,
            *(
                text
                for section in self.sections
                for block in section.blocks
                for text in (
                    (block.text,)
                    if block.source is None
                    else (
                        block.source.title,
                        display_url(block.source.url),
                        block.source.snippet[:200],
                    )
                )
                if block.kind != "chart"
            ),
            *(section.title for section in self.sections),
        ]


def _findings(report: ResearchReport) -> list[Section]:
    verdicts = report.finding_verifications or []
    findings = [
        Block("notice", LEGACY_VERDICT_NOTICE)
        if verdicts
        else Block("notice", UNVERIFIED_NOTICE)
    ]
    for number, finding in enumerate(report.key_findings, 1):
        verdict = verdicts[number - 1] if number - 1 < len(verdicts) else None
        tag = f"  {legacy_verdict_tag(verdict)}" if verdict is not None else ""
        findings.append(Block("text", f"{number}. {finding}{tag}"))
    sections = [
        Section("Summary", [Block("text", report.summary)]),
        Section("Key Findings", findings if report.key_findings else []),
    ]
    if report.analysis:
        sections.append(Section("Analysis", [Block("text", report.analysis)]))
    return sections


def _evidence_review(report: ResearchReport) -> list[Section]:
    questions = report.research_plan.get("questions", [])
    if report.claims or not questions:
        return []
    blocks = [Block("notice", REVIEW_NOTICE)]
    for item in questions:
        blocks.append(Block("text", f"### {item['question_id']}: {item['question']}"))
        review = item.get("review")
        if review:
            blocks.extend(
                [
                    Block("text", f"Status: {review['status']}"),
                    Block("text", review["conclusion"]),
                    Block(
                        "text",
                        "Counterevidence investigation: "
                        + review["counterevidence_note"],
                    ),
                    *(
                        Block(
                            "item",
                            f"{label} passages: "
                            + (", ".join(review[key]) or "none recorded"),
                        )
                        for key, label in (
                            ("supporting_passage_ids", "Supporting"),
                            ("challenging_passage_ids", "Challenging"),
                        )
                    ),
                ]
            )
        blocks.extend(Block("item", f"Gap: {gap}") for gap in item["gaps"])
    return [Section("Evidence Review", blocks)]


def build_document(report: ResearchReport) -> ReportDocument:
    if report.status not in {"ok", "partial"}:
        heading = (
            "Insufficient evidence"
            if report.status == "insufficient_evidence"
            else "Research failed"
        )
        return ReportDocument(
            title=f"{heading}: {report.query}",
            completed=False,
            sections=[
                Section(
                    "",
                    [Block("text", report.error or "No report was produced.")],
                )
            ],
            search_queries=report.search_queries,
        )
    sections = (
        [
            Section(
                CLAIM_SECTION_TITLE,
                [
                    Block("notice", CLAIM_SECTION_NOTICE),
                    Block("claims", claim_report_body(report)),
                ],
            )
        ]
        if report.claims
        else _findings(report)
    )
    for title, text in (
        ("Source Cross-Analysis", report.cross_analysis),
        ("Uncertainties and Outlook", report.predictions),
        ("Recorded Confidence", report.confidence),
    ):
        if text:
            sections.append(Section(title, [Block("text", text)]))
    sections.extend(_evidence_review(report))
    sections.append(
        Section(
            "Research Limitations",
            [
                Block("text", f"Verification scope: {report.verification_scope}"),
                *(Block("item", note) for note in report.limitations),
            ],
        )
    )
    if report.market_data_summary or report.charts:
        sections.append(
            Section(
                "Market Data",
                [
                    *(
                        [Block("text", report.market_data_summary)]
                        if report.market_data_summary
                        else []
                    ),
                    *(Block("chart", path) for path in report.charts),
                ],
            )
        )
    for title, note, sources in reference_groups(report):
        sections.append(
            Section(
                f"{title} ({len(sources)})",
                [
                    Block("notice", note),
                    *(
                        Block("reference", number=number, source=source)
                        for number, source in sources
                    ),
                ],
            )
        )
    return ReportDocument(
        title=report.query,
        completed=True,
        generated=f"Generated {report.timestamp.strftime('%Y-%m-%d %H:%M')}"
        f" | Model: {report.model_used} | Sources: {len(report.sources)}",
        notice=PARTIAL_NOTICE if report.status == "partial" else "",
        sections=sections,
        search_queries=report.search_queries,
    )


def _reference_lines(block: Block, markdown: bool) -> list[str]:
    source = block.source
    assert source is not None
    if not markdown:
        lines = [
            f"- [Source {block.number}] [{source.title}]({display_url(source.url)})"
        ]
        evidence = source.supporting_snippet or source.snippet
        return lines + ([f"   {evidence}"] if evidence else [])
    lines = [
        f"- **[Source {block.number}] {source.title}**  ",
        f"  {display_url(source.url)}",
    ]
    if source.retrieved_at:
        lines.append(
            f"  Retrieved: {source.retrieved_at}; "
            f"published: {source.published_at or 'unknown'}"
        )
    if source.source_kind:
        lines.append(f"  Source-type hint: {source.source_kind}")
    if source.passage_ids:
        lines.append(f"  Passages: {', '.join(source.passage_ids)}")
    if source.content_hash:
        lines.append(f"  Content SHA-256: {source.content_hash}")
    return lines


def render_text(document: ReportDocument, markdown: bool) -> str:
    """Markdown, or plain text that keeps the host's own Markdown as written.

    Only Markdown output escapes claim quotes, because only a Markdown viewer
    would interpret the untrusted source text inside them.
    """
    lines = [f"# {document.title}", ""]
    if document.completed:
        lines.extend([f"*{document.generated}*", "", "---", ""])
    if document.notice:
        lines.extend([f"> {document.notice}", ""])
    for section in document.sections:
        if section.title:
            lines.extend([f"## {section.title}", ""])
        previous = ""
        for block in section.blocks:
            if previous in {"item", "reference"} and block.kind != previous:
                lines.append("")
            if block.kind == "item":
                lines.append(f"- {block.text}")
            elif block.kind == "reference":
                lines.extend(_reference_lines(block, markdown))
            elif block.kind == "notice":
                lines.extend([f"> {block.text}", ""])
            elif block.kind == "claims":
                lines.extend(
                    [escape_quote_lines(block.text) if markdown else block.text, ""]
                )
            elif block.kind == "chart":
                lines.extend([f"Chart: {block.text}", ""])
            else:
                lines.extend([block.text, ""])
            previous = block.kind
        if previous in {"item", "reference"}:
            lines.append("")
    if document.search_queries:
        lines.extend(
            ["---", "", f"*Search queries: {' | '.join(document.search_queries)}*"]
        )
    return "\n".join(lines).rstrip("\n") + "\n"
