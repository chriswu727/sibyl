"""Shared presentation of evidence-linked report conclusions."""

import re

from .report_models import FindingVerification, ResearchReport

CLAIM_SECTION_TITLE = "Evidence-linked conclusions"
CLAIM_SECTION_NOTICE = (
    "Host-declared facts, inferences, recommendations and unknowns. "
    "Citation checks do not independently verify semantic support."
)
LEGACY_VERDICT_NOTICE = (
    "Historical recorded verdicts; not revalidated on read or export. "
    "Recorded support and confidence are not independent verification."
)


def legacy_verdict_tag(verdict: FindingVerification) -> str:
    support = "supported" if verdict.supported else "unverified"
    return f"[historical: {support}; recorded confidence: {verdict.confidence}]"


# Quotes are untrusted source text and always sit mid-line after this prefix, so
# only inline Markdown (emphasis, code, links, images, HTML, entities) can apply.
_QUOTE_LINE = re.compile(r'^(Quote \[Source \d+\]: ")(.*)(")$')
_MARKDOWN_INLINE = re.compile(r"([\\`*_\[\]<>!~|&])")


def markdown_literal(text: str) -> str:
    """Escape inline Markdown so source text renders as written."""
    return _MARKDOWN_INLINE.sub(r"\\\1", text)


def escape_quote_lines(body: str) -> str:
    """Escape the quoted source text in each claim quote line for Markdown."""
    return "\n".join(
        match[1] + markdown_literal(match[2]) + match[3]
        if (match := _QUOTE_LINE.match(line))
        else line
        for line in body.split("\n")
    )


def claim_report_body(report: ResearchReport, markdown: bool = False) -> str:
    body = report.analysis or "\n\n".join(
        dict.fromkeys([report.summary, *report.key_findings])
    )
    return escape_quote_lines(body) if markdown and report.claims else body
