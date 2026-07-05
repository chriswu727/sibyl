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
    fv = report.finding_verifications or []
    for i, finding in enumerate(report.key_findings, 1):
        v = fv[i - 1] if i - 1 < len(fv) else None
        tag = ""
        if v is not None:
            tag = f"  _[confidence: {v.confidence}]_" if v.supported else "  _(unverified)_"
        lines.append(f"{i}. {finding}{tag}")
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

    lines.append("## References")
    lines.append("")
    for i, src in enumerate(report.sources, 1):
        lines.append(f"{i}. **{src.title}**  ")
        lines.append(f"   {src.url}")
        evidence = src.supporting_snippet or src.snippet
        if evidence:
            lines.append(f"   > {evidence}")
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

    def _plain(line, size):
        # Last-resort render if markdown parsing raises: strip the syntax.
        import re
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        clean = re.sub(r'^#{1,6}\s*', '', clean)
        try:
            pdf.set_font(font_name, "", size)
            pdf.multi_cell(0, 6, clean)
        except Exception:
            pass

    def body(text, size=10):
        """Render markdown text preserving structure: ## headers become sub-
        headings, bullets keep their marker, and **bold** renders inline (the LLM
        emits real markdown — the old renderer flattened all of it to plain text)."""
        import re
        if not text:
            return
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line.strip():
                pdf.ln(2)
                continue
            stripped = line.strip()
            hm = re.match(r'^(#{1,6})\s+(.*)$', stripped)
            if hm:
                hsize = {1: 15, 2: 13, 3: 12}.get(len(hm.group(1)), 11)
                try:
                    pdf.set_font(font_name, "B", hsize)
                    pdf.multi_cell(0, 6, hm.group(2), markdown=True)
                    pdf.ln(1)
                except Exception:
                    _plain(stripped, size)
                continue
            bm = re.match(r'^[-*•]\s+(.*)$', stripped)
            if bm:
                try:
                    pdf.set_font(font_name, "", size)
                    pdf.multi_cell(0, 6, "   •  " + bm.group(1), markdown=True)
                except Exception:
                    _plain(stripped, size)
                continue
            try:
                pdf.set_font(font_name, "", size)
                pdf.multi_cell(0, 6, line, markdown=True)
            except Exception:
                _plain(line, size)
        pdf.ln(2)

    def separator():
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    # Title
    heading(report.query, 18)
    pdf.set_font(font_name, "", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, f"Generated {report.timestamp.strftime('%Y-%m-%d %H:%M')} | Model: {report.model_used} | Sources: {len(report.sources)}")
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)
    separator()

    # Table of Contents
    heading("Table of Contents", 12)
    toc_items = ["1. Summary", "2. Key Findings"]
    if report.analysis:
        toc_items.append("3. Analysis")
    if report.cross_analysis:
        toc_items.append(f"{len(toc_items)+1}. Source Cross-Analysis")
    if report.predictions:
        toc_items.append(f"{len(toc_items)+1}. Predictions")
    if report.market_data_summary or report.charts:
        toc_items.append(f"{len(toc_items)+1}. Market Data")
    toc_items.append(f"{len(toc_items)+1}. Sources ({len(report.sources)})")
    for item in toc_items:
        pdf.set_font(font_name, "", 10)
        try:
            pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")
        except Exception:
            pass
    pdf.ln(4)
    separator()

    # Summary
    heading("Summary", 14)
    body(report.summary)
    separator()

    # Key Findings — annotate each with its verification verdict (if present).
    heading("Key Findings", 14)
    fv = report.finding_verifications or []
    for i, finding in enumerate(report.key_findings, 1):
        v = fv[i - 1] if i - 1 < len(fv) else None
        tag = ""
        if v is not None:
            tag = f"  [confidence: {v.confidence}]" if v.supported else "  (unverified)"
        body(f"{i}. {finding}{tag}")
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
    heading(f"Sources ({len(report.sources)})", 14)
    for i, src in enumerate(report.sources, 1):
        try:
            pdf.set_font(font_name, "B", 9)
            pdf.multi_cell(0, 5, f"{i}. {src.title}")
        except Exception:
            pass
        try:
            pdf.set_font(font_name, "", 7)
            pdf.set_text_color(30, 30, 180)
            pdf.write(4, src.url)
            pdf.ln(4)
            pdf.set_text_color(0, 0, 0)
        except Exception:
            pass
        if src.snippet:
            try:
                pdf.set_font(font_name, "", 7)
                pdf.set_text_color(100, 100, 100)
                pdf.write(4, src.snippet[:200])
                pdf.ln(4)
                pdf.set_text_color(0, 0, 0)
            except Exception:
                pass
        pdf.ln(3)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sibyl_{_safe_filename(report.query)}_{ts}.pdf"
    path = out / filename
    pdf.output(str(path))
    return str(path)
