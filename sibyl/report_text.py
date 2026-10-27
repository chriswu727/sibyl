"""Plain-text presentation of research reports for CLI and MCP clients."""

from .report_document import build_document, render_text
from .report_models import ResearchReport


def _format_report(report: ResearchReport) -> str:
    """Format a research report as readable text."""
    return render_text(build_document(report), markdown=False)
