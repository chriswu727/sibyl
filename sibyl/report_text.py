"""Plain-text presentation of research reports for CLI and MCP clients."""

from .privacy import display_url
from .report_models import ResearchReport


def _format_report(report: ResearchReport) -> str:
    """Format a research report as readable text."""
    if report.status != "ok":
        heading = (
            "Insufficient evidence"
            if report.status == "insufficient_evidence"
            else "Research failed"
        )
        lines = [
            f"# {heading}: {report.query}",
            "",
            "No report was produced."
            if report.status == "failed"
            else (report.error or "No report was produced."),
        ]
        if report.search_queries:
            lines.extend(
                ["", f"*Search queries attempted: {', '.join(report.search_queries)}*"]
            )
        return "\n".join(lines)

    lines = [
        f"# Research Report: {report.query}",
        f"*Generated at {report.timestamp.strftime('%Y-%m-%d %H:%M')} using {report.model_used}*",
        "",
        "## Summary",
        report.summary,
        "",
        "## Key Findings",
    ]
    fv = report.finding_verifications or []
    if report.key_findings and not fv:
        lines.append("> Claim verification was not performed for this report.")
    for i, finding in enumerate(report.key_findings, 1):
        verdict = fv[i - 1] if i - 1 < len(fv) else None
        tag = ""
        if verdict is not None:
            tag = (
                f"  [verified: {verdict.confidence}]"
                if verdict.supported
                else "  (unverified)"
            )
        lines.append(f"{i}. {finding}{tag}")

    if report.analysis:
        lines.append("")
        lines.append("## Analysis")
        lines.append(report.analysis)

    if report.predictions:
        lines.append("")
        lines.append("## Uncertainties and Outlook")
        lines.append(report.predictions)

    if report.confidence:
        lines.append("")
        lines.append(f"**Confidence:** {report.confidence}")

    if report.cross_analysis:
        lines.append("")
        lines.append(report.cross_analysis)

    lines.extend(
        [
            "",
            "## Research Limitations",
            f"Verification scope: {report.verification_scope}",
        ]
    )
    lines.extend(f"- {note}" for note in report.limitations)

    lines.append("")
    lines.append(f"## Sources ({len(report.sources)})")
    for i, src in enumerate(report.sources, 1):
        lines.append(f"{i}. [{src.title}]({display_url(src.url)})")
        evidence = src.supporting_snippet or src.snippet
        if evidence:
            lines.append(f"   {evidence}")

    if report.market_data_summary:
        lines.append("")
        lines.append(report.market_data_summary)

    lines.append("")
    lines.append(f"*Search queries used: {', '.join(report.search_queries)}*")
    return "\n".join(lines)
