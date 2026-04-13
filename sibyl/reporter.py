"""Report generator — creates PDF and Markdown reports from research results."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .researcher import ResearchReport


def _report_to_markdown(report: ResearchReport) -> str:
    """Convert a ResearchReport to markdown text."""
    lines = [
        f"# {report.query}",
        "",
        f"*Generated {report.timestamp.strftime('%Y-%m-%d %H:%M')} | Model: {report.model_used}*",
        "",
        "---",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Key Findings",
        "",
    ]
    for i, finding in enumerate(report.key_findings, 1):
        lines.append(f"{i}. {finding}")
    lines.append("")

    if report.analysis:
        lines.append("## Analysis")
        lines.append("")
        lines.append(report.analysis)
        lines.append("")

    if report.predictions:
        lines.append("## Predictions")
        lines.append("")
        lines.append(report.predictions)
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    for i, src in enumerate(report.sources, 1):
        lines.append(f"{i}. **{src.title}**  ")
        lines.append(f"   {src.url}")
    lines.append("")

    if report.search_queries:
        lines.append("---")
        lines.append("")
        lines.append(f"*Search queries: {' | '.join(report.search_queries)}*")

    return "\n".join(lines)


def _safe_filename(query: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "" for c in query)[:50].strip()


def generate_pdf(report: ResearchReport, output_dir: str = ".") -> str:
    """Generate a PDF report using fpdf2 (pure Python, no system deps)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Try to use a Unicode font, fall back to Helvetica
    try:
        pdf.add_font("NotoSans", "", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", uni=True)
        pdf.add_font("NotoSans", "B", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", uni=True)
        font_name = "NotoSans"
    except Exception:
        font_name = "Helvetica"

    def heading(text, size=18, style="B"):
        pdf.set_font(font_name, style, size)
        pdf.multi_cell(0, 8, text)
        pdf.ln(2)

    def body(text, size=10):
        pdf.set_font(font_name, "", size)
        pdf.multi_cell(0, 6, text)
        pdf.ln(2)

    def separator():
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    # Title
    heading(report.query, 18)
    pdf.set_font(font_name, "", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, f"Generated {report.timestamp.strftime('%Y-%m-%d %H:%M')} | Model: {report.model_used}")
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)
    separator()

    # Summary
    heading("Summary", 14)
    body(report.summary)
    separator()

    # Key Findings
    heading("Key Findings", 14)
    for i, finding in enumerate(report.key_findings, 1):
        body(f"{i}. {finding}")
    separator()

    # Analysis
    if report.analysis:
        heading("Analysis", 14)
        body(report.analysis)
        separator()

    # Predictions
    if report.predictions:
        heading("Predictions", 14)
        body(report.predictions)
        separator()

    # Sources
    heading("Sources", 14)
    for i, src in enumerate(report.sources, 1):
        pdf.set_font(font_name, "B", 9)
        pdf.multi_cell(0, 5, f"{i}. {src.title[:80]}")
        pdf.set_font(font_name, "", 7)
        pdf.set_text_color(50, 50, 200)
        url_display = src.url if len(src.url) < 90 else src.url[:87] + "..."
        pdf.cell(0, 4, url_display, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sibyl_{_safe_filename(report.query)}_{ts}.pdf"
    path = out / filename
    pdf.output(str(path))
    return str(path)
