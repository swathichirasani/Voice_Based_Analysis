from __future__ import annotations

from pathlib import Path
from typing import Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_pdf_report(
    output_path: str | Path,
    reference: str,
    transcript: str,
    waveform_path: str | Path,
    metrics: Dict[str, object],
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Reference Concept</b>", styles["Heading2"]))
    story.append(Paragraph(reference, styles["BodyText"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("<b>Student Transcription</b>", styles["Heading2"]))
    story.append(Paragraph(transcript, styles["BodyText"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("<b>Audio Visualization</b>", styles["Heading2"]))
    if Path(waveform_path).exists():
        story.append(Image(str(waveform_path), width=5.8 * inch, height=2.0 * inch))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("<b>Evaluation Summary</b>", styles["Heading2"]))
    rows = [["Metric", "Value"]]
    rows.extend(
        [
            ["Semantic Similarity", metrics.get("semantic_similarity", "")],
            ["Filler Word Ratio", metrics.get("filler_ratio", "")],
            ["Pause Ratio", metrics.get("pause_ratio", "")],
            ["Confidence (Energy)", metrics.get("rms_energy", "")],
            ["Final Score", f"{metrics.get('score', '')}/100"],
            ["Understanding Level", metrics.get("level", "")],
        ]
    )

    table = Table(rows, colWidths=[2.5 * inch, 3.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "Qualitative Feedback: The report combines concept similarity, filler usage, "
            "pause ratio, and signal energy to provide a transparent understanding estimate.",
            styles["BodyText"],
        )
    )

    doc.build(story)
    return str(output)
