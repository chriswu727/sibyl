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
        try:
            pdf.set_font(font_name, style, size)
            pdf.multi_cell(0, 8, text[:200])
            pdf.ln(2)
        except Exception:
            pass

    def body(text, size=10):
        try:
            import re
            # Strip markdown formatting for PDF
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** → bold
            clean = re.sub(r'\*(.+?)\*', r'\1', clean)      # *italic* → italic
            clean = re.sub(r'^#{1,4}\s*', '', clean, flags=re.MULTILINE)  # ## headers → plain
            clean = re.sub(r'^\d+\.\s*\d+\.', lambda m: m.group().split('.')[0] + '.', clean, flags=re.MULTILINE)  # "1. 1." → "1."
            pdf.set_font(font_name, "", size)
            pdf.multi_cell(0, 6, clean)
            pdf.ln(2)
        except Exception:
            pass

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

    # Key Findings (strip duplicate numbering like "1. 1.")
    heading("Key Findings", 14)
    for i, finding in enumerate(report.key_findings, 1):
        body(f"{i}. {finding}")
    separator()

    # Analysis
    if report.analysis:
        heading("Analysis", 14)
        body(report.analysis)
        separator()

    # Cross-analysis
    if report.cross_analysis:
        heading("Source Cross-Analysis", 14)
        import re
        for line in report.cross_analysis.splitlines():
            clean = re.sub(r'[#]', '', line).strip()
            # Skip duplicate heading
            if clean.lower().replace("*", "").strip().startswith("source cross"):
                continue
            if clean:
                if clean.startswith("**") and clean.endswith("**"):
                    heading(clean.strip("* "), 11)
                else:
                    body(clean, 9)

    # Predictions
    if report.predictions:
        heading("Predictions", 14)
        body(report.predictions)
        separator()

    # Market Data + Charts
    if report.market_data_summary or report.charts:
        heading("Market Data", 14)
        if report.market_data_summary:
            import re
            clean_data = re.sub(r'[#*]', '', report.market_data_summary)
            for line in clean_data.strip().splitlines():
                line = line.strip()
                if line:
                    try:
                        pdf.set_font(font_name, "", 9)
                        pdf.multi_cell(0, 5, line)
                    except Exception:
                        pass
            pdf.ln(4)

        for chart_path in report.charts:
            if Path(chart_path).exists():
                # Calculate image width to fit page
                page_w = pdf.w - pdf.l_margin - pdf.r_margin
                try:
                    pdf.image(chart_path, x=pdf.l_margin, w=page_w)
                    pdf.ln(6)
                except Exception:
                    body(f"[Chart: {chart_path}]")
        separator()

    # Sources
    heading("Sources", 14)
    for i, src in enumerate(report.sources, 1):
        pdf.set_font(font_name, "B", 9)
        pdf.multi_cell(0, 5, f"{i}. {src.title[:80]}")
        pdf.set_font(font_name, "", 7)
        pdf.set_text_color(50, 50, 200)
        try:
            pdf.multi_cell(0, 4, src.url)
        except Exception:
            pdf.cell(0, 4, src.url[:80] + "...", new_x="LMARGIN", new_y="NEXT")
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
