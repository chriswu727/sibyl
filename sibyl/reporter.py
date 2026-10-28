"""Report generator — creates PDF and Markdown reports from research results."""

from __future__ import annotations

from pathlib import Path

from .privacy import display_url
from .report_document import build_document, render_text
from .report_models import ResearchReport
from .storage import ArtifactStore


class PDFRenderingError(ValueError):
    pass


def _report_to_markdown(report: ResearchReport) -> str:
    """Convert a ResearchReport to markdown text."""
    return render_text(build_document(report), markdown=True)


def _safe_filename(query: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "" for c in query)[:50].strip()


def generate_pdf(report: ResearchReport, output_dir: str = ".") -> str:
    """Generate a PDF with explicit font-coverage and required-rendering failures."""
    if report.status not in {"ok", "partial"}:
        raise ValueError(
            report.error or "Cannot generate a PDF without a completed report"
        )
    from fpdf import FPDF

    pdf = FPDF()
    pdf.core_fonts_encoding = "cp1252"
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Try to use a Unicode font, fall back to Helvetica
    font_name = "Helvetica"
    for regular, bold in (
        (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ),
    ):
        if not Path(regular).is_file() or not Path(bold).is_file():
            continue
        try:
            pdf.add_font("ReportUnicode", "", regular)
            pdf.add_font("ReportUnicode", "B", bold)
            font_name = "ReportUnicode"
            break
        except Exception:
            continue

    document = build_document(report)
    required_text = document.texts()
    pdf.set_font(font_name, "", 10)
    cmap = getattr(pdf.current_font, "cmap", None)
    if cmap is not None:
        unsupported = bool(
            {ord(char) for text in required_text for char in text if not char.isspace()}
            - cmap.keys()
        )
    else:
        try:
            "\n".join(required_text).encode(pdf.core_fonts_encoding)
            unsupported = False
        except UnicodeEncodeError:
            unsupported = True
    if unsupported:
        raise PDFRenderingError(
            "PDF font cannot render required report text; use Markdown/JSON or a compatible Unicode font."
        )

    def heading(text, size=18, style="B"):
        try:
            pdf.set_font(font_name, style, size)
            pdf.multi_cell(0, 8, text[:200], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        except Exception as exc:
            raise PDFRenderingError("PDF could not render a report heading.") from exc

    def _plain(line, size):
        # Last-resort render if markdown parsing raises: strip the syntax.
        import re

        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        clean = re.sub(r"^#{1,6}\s*", "", clean)
        try:
            pdf.set_font(font_name, "", size)
            pdf.multi_cell(0, 6, clean, new_x="LMARGIN", new_y="NEXT")
        except Exception as exc:
            raise PDFRenderingError(
                "PDF could not render required report text; use Markdown/JSON or a compatible Unicode font."
            ) from exc

    def body(text, size=10):
        """Render markdown text preserving structure: ## headers become sub-
        headings, bullets keep their marker, and **bold** renders inline."""
        import re

        if not text:
            return
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line.strip():
                pdf.ln(2)
                continue
            stripped = line.strip()
            hm = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if hm:
                hsize = {1: 15, 2: 13, 3: 12}.get(len(hm.group(1)), 11)
                try:
                    pdf.set_font(font_name, "B", hsize)
                    pdf.multi_cell(
                        0, 6, hm.group(2), markdown=True, new_x="LMARGIN", new_y="NEXT"
                    )
                    pdf.ln(1)
                except Exception:
                    _plain(stripped, size)
                continue
            bm = re.match(r"^[-*•]\s+(.*)$", stripped)
            if bm:
                try:
                    pdf.set_font(font_name, "", size)
                    pdf.multi_cell(
                        0,
                        6,
                        "   -  " + bm.group(1),
                        markdown=True,
                        new_x="LMARGIN",
                        new_y="NEXT",
                    )
                except Exception:
                    _plain(stripped, size)
                continue
            try:
                pdf.set_font(font_name, "", size)
                pdf.multi_cell(0, 6, line, markdown=True, new_x="LMARGIN", new_y="NEXT")
            except Exception:
                _plain(line, size)
        pdf.ln(2)

    def separator():
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    heading(document.title, 18)
    if document.notice:
        body(document.notice)
    pdf.set_font(font_name, "", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, document.generated)
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)
    separator()

    # The table of contents lists exactly the sections rendered below.
    heading("Table of Contents", 12)
    for number, section in enumerate(document.sections, 1):
        pdf.set_font(font_name, "", 10)
        try:
            pdf.cell(0, 6, f"{number}. {section.title}", new_x="LMARGIN", new_y="NEXT")
        except Exception as exc:
            raise PDFRenderingError(
                "PDF could not render the table of contents."
            ) from exc
    pdf.ln(4)
    separator()

    def reference(i, src):
        try:
            pdf.set_font(font_name, "B", 9)
            pdf.multi_cell(
                0, 5, f"[Source {i}] {src.title}", new_x="LMARGIN", new_y="NEXT"
            )
        except Exception as exc:
            raise PDFRenderingError("PDF could not render a source title.") from exc
        try:
            pdf.set_font(font_name, "", 7)
            pdf.set_text_color(30, 30, 180)
            pdf.write(4, display_url(src.url))
            pdf.ln(4)
            pdf.set_text_color(0, 0, 0)
        except Exception as exc:
            raise PDFRenderingError("PDF could not render a source URL.") from exc
        if src.snippet:
            try:
                pdf.set_font(font_name, "", 7)
                pdf.set_text_color(100, 100, 100)
                pdf.write(4, src.snippet[:200])
                pdf.ln(4)
                pdf.set_text_color(0, 0, 0)
            except Exception as exc:
                raise PDFRenderingError(
                    "PDF could not render a source excerpt."
                ) from exc
        pdf.ln(3)

    for section in document.sections:
        heading(section.title, 14)
        for block in section.blocks:
            if block.kind == "reference":
                reference(block.number, block.source)
            elif block.kind == "item":
                body(f"- {block.text}")
            elif block.kind == "chart":
                try:
                    pdf.image(
                        block.text,
                        x=pdf.l_margin,
                        w=pdf.w - pdf.l_margin - pdf.r_margin,
                    )
                    pdf.ln(6)
                except Exception:
                    body(f"[Chart: {block.text}]")
            else:
                body(block.text)
        separator()

    # Save
    return str(ArtifactStore(output_dir).write(bytes(pdf.output()), "report", ".pdf"))
