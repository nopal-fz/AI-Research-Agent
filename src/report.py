from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER


def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    w, h = A4
    canvas.drawString(20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf(topic: str, result: dict) -> bytes:
    """Generate a polished PDF report from the structured `result` dict.

    `result` may contain keys: `final_report`, `sub_questions`, `search_queries`,
    `source_summaries`, and `research_metrics`. The function builds a readable,
    justified PDF with spacing and page numbers.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=f"Research Report - {topic}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )

    list_style = ParagraphStyle(
        "List",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        leftIndent=12,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )

    elems = []
    elems.append(Paragraph(f"Research Report", title_style))
    elems.append(Paragraph(topic, heading_style))
    elems.append(Spacer(1, 6))

    # Metrics
    metrics = result.get("research_metrics") or {}
    if metrics:
        elems.append(Paragraph("Metrics", heading_style))
        metrics_lines = "<br/>".join([f"<b>{k}:</b> {v}" for k, v in metrics.items()])
        elems.append(Paragraph(metrics_lines, body_style))

    # Sub-questions
    sub_q = result.get("sub_questions") or []
    if sub_q:
        elems.append(Paragraph("Sub-questions", heading_style))
        for q in sub_q:
            elems.append(Paragraph(f"• {q}", list_style))

    # Search queries
    searches = result.get("search_queries") or []
    if searches:
        elems.append(Paragraph("Search Queries", heading_style))
        for s in searches:
            elems.append(Paragraph(f"• {s}", list_style))

    # helper: strip simple markdown markers from text
    def _clean_markdown(text: str) -> str:
        import re

        if not text:
            return ""
        # remove code fences
        text = re.sub(r"```.+?```", "", text, flags=re.DOTALL)
        # remove headings like #, ##, ###
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        # remove emphasis markers *, **, _, __, `code`
        text = re.sub(r"\*\*|\*|__|_|", "", text)
        text = re.sub(r"`+", "", text)
        # remove blockquote markers
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
        return text.strip()

    # Final report / summary
    final = _clean_markdown(result.get("final_report") or "")
    if final:
        elems.append(Paragraph("Final Report", heading_style))
        for para in final.split("\n\n"):
            elems.append(Paragraph(para.replace("\n", " "), body_style))

    # Source summaries (optional)
    sources = result.get("source_summaries") or []
    if sources:
        elems.append(Paragraph("Source Summaries", heading_style))
        for s in sources:
            # each source may be dict or str
            if isinstance(s, dict):
                title = s.get("title") or s.get("source") or "Source"
                text = s.get("summary") or s.get("text") or ""
                elems.append(Paragraph(f"<b>{title}</b>", list_style))
                elems.append(Paragraph(text.replace("\n", " "), body_style))
            else:
                elems.append(Paragraph(s.replace("\n", " "), body_style))

    doc.build(elems, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
