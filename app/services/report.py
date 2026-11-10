"""PDF forensic report generation (ReportLab)."""

from __future__ import annotations

import base64
import io
from datetime import UTC, datetime

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.schemas import AnalysisResult

VERDICT_COLORS = {"fake": "#d6453d", "real": "#1f9d6b", "uncertain": "#d9930d"}
VERDICT_TEXT = {
    "fake": "LIKELY MANIPULATED",
    "real": "LIKELY AUTHENTIC",
    "uncertain": "INCONCLUSIVE",
}
NOTE_TEXT = {
    "no_face": "No face was detected, so the whole frame was analyzed. Accuracy is lower in this mode.",
    "multiple_faces": "Several faces were found. The overall score is the highest face score.",
    "partial_faces": "A face was found in only some frames; frames without a face are skipped.",
}


KIND_TEXT = {
    "ffhq": "FFHQ photo", "stylegan": "StyleGAN face", "imdb_wiki": "celebrity photo",
    "sd_text2img": "Stable Diffusion image", "sd_inpainting": "diffusion edit", "insightface": "InsightFace swap",
    "celebdf_real": "Celeb-DF real", "celebdf_swap": "Celeb-DF swap", "dfd_real": "DFD real", "dfd_swap": "DFD swap",
}


def _img(b64: str | None, width: float) -> Image | str:
    if not b64:
        return ""
    raw = io.BytesIO(base64.b64decode(b64))
    img = Image(raw)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width, width * ratio
    return img


def _timeline_chart(result: AnalysisResult) -> Drawing | None:
    pts = [(p.t, p.fake_probability) for p in result.timeline if p.fake_probability is not None]
    if len(pts) < 2:
        return None
    d = Drawing(170 * mm, 60 * mm)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 12 * mm, 10 * mm, 150 * mm, 42 * mm
    lp.data = [pts]
    lp.lines[0].strokeColor = colors.HexColor("#3b5bdb")
    lp.lines[0].strokeWidth = 1.6
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax, lp.yValueAxis.valueStep = 0, 1, 0.25
    lp.xValueAxis.labelTextFormat = "%.0fs"
    d.add(lp)
    y = lp.y + lp.height * 0.5
    d.add(Line(lp.x, y, lp.x + lp.width, y, strokeColor=colors.HexColor("#999999"), strokeDashArray=[3, 3]))
    d.add(String(lp.x, lp.y + lp.height + 4, "Fake probability per sampled frame", fontSize=8))
    return d


def build_pdf(result: AnalysisResult) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"DeepShield report {result.id}")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=2, alignment=0)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#555555"))
    color = colors.HexColor(VERDICT_COLORS[result.verdict])
    big = ParagraphStyle("big", parent=styles["Heading1"], textColor=color, fontSize=22, spaceAfter=4)

    story = [
        Paragraph("DeepShield Forensic Report", h1),
        Paragraph(
            f"Report ID {result.id} &nbsp;|&nbsp; {datetime.now(UTC):%Y-%m-%d %H:%M UTC}"
            f" &nbsp;|&nbsp; File: {result.filename}", small),
        Spacer(1, 8 * mm),
        Paragraph(VERDICT_TEXT[result.verdict], big),
        Paragraph(
            f"Fake probability <b>{result.fake_probability:.1%}</b> &nbsp; Confidence "
            f"<b>{result.confidence:.0%}</b>", styles["Normal"]),
        Spacer(1, 6 * mm),
    ]

    meta = [
        ["Media type", result.media_type],
        ["Frames analyzed", str(result.frames_analyzed)],
        ["Scoring engine", result.engine],
        ["Processing time", f"{result.processing_ms} ms"],
    ]
    if result.duration_seconds:
        meta.insert(1, ["Duration", f"{result.duration_seconds:.1f} s"])
    t = Table(meta, colWidths=[45 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t, Spacer(1, 6 * mm)]

    for i, face in enumerate(result.faces[:4], start=1):
        label = "Most suspicious frame" if result.media_type == "video" else f"Face {i}"
        story.append(Paragraph(f"<b>{label}</b> : {face.fake_probability:.1%} fake", styles["Heading3"]))
        imgs = Table([[_img(face.crop_png, 55 * mm), _img(face.heatmap_png, 55 * mm)],
                      ["Input crop", "Grad-CAM evidence map"]], colWidths=[62 * mm, 62 * mm])
        imgs.setStyle(TableStyle([("FONTSIZE", (0, 1), (-1, 1), 8), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        rows = [["Backbone", "Fake probability"]] + [[k, f"{v:.1%}"] for k, v in face.model_scores.items()]
        scores = Table(rows, colWidths=[90 * mm, 40 * mm])
        scores.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f3f5")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ]))
        story += [imgs, Spacer(1, 2 * mm), scores, Spacer(1, 2 * mm)]
        if face.similar:
            fakes = sum(s.label == "fake" for s in face.similar)
            kinds = ", ".join(f"{KIND_TEXT.get(s.kind, s.kind)} ({s.label}, {s.similarity:.2f})" for s in face.similar)
            story.append(Paragraph(
                f"<b>Closest known cases:</b> {fakes} of {len(face.similar)} most similar training faces are fakes. "
                f"{kinds}.", small))
        story.append(Spacer(1, 5 * mm))

    chart = _timeline_chart(result)
    if chart:
        story += [Paragraph("<b>Timeline</b>", styles["Heading3"]), chart, Spacer(1, 4 * mm)]

    for note in result.notes:
        if note in NOTE_TEXT:
            story.append(Paragraph(f"Note: {NOTE_TEXT[note]}", small))

    story += [
        Spacer(1, 6 * mm),
        Paragraph(
            "Disclaimer: this report is produced by a machine-learning model and is probabilistic. "
            "It can be wrong, especially on heavily compressed media or manipulation methods not seen "
            "during training. Treat it as one piece of evidence, not as proof.", small),
    ]
    doc.build(story)
    return buf.getvalue()
