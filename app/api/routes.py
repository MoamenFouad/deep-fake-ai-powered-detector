"""HTTP endpoints."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.api.deps import get_analyzer, is_loaded, results
from app.config import ROOT_DIR, settings
from app.schemas import AnalysisResult, HealthResponse
from app.services.imaging import InvalidMediaError
from app.services.report import build_pdf

router = APIRouter(prefix="/api")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"


async def _read_limited(file: UploadFile, limit_mb: float) -> bytes:
    limit = int(limit_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File too large. Maximum is {limit_mb:g} MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    return data


def _kind(file: UploadFile) -> str | None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    ctype = (file.content_type or "").lower()
    if ctype in settings.image_types or ext in IMAGE_EXT:
        return "image"
    if ctype in settings.video_types or ext in VIDEO_EXT:
        return "video"
    return None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = is_loaded()
    analyzer = get_analyzer() if loaded else None
    return HealthResponse(
        status="ok",
        version=settings.version,
        models_loaded=loaded,
        engine=analyzer.classifier.engine if analyzer else "loading",
        backbones=list(analyzer.classifier.names) if analyzer else settings.model_ids,
        similar_cases=bool(analyzer and analyzer.similar is not None),
    )


@router.get("/metrics", summary="Held-out test metrics from the latest training run")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics yet. Run ml/scripts/train_fusion.py.")
    return json.loads(METRICS_PATH.read_text())


@router.post("/analyze", response_model=AnalysisResult, summary="Analyze an image or a video")
async def analyze(file: UploadFile = File(...)) -> AnalysisResult:
    kind = _kind(file)
    if kind is None:
        raise HTTPException(415, "Unsupported file type. Upload JPG, PNG, WEBP, MP4, MOV, AVI, WEBM or MKV.")
    limit = settings.max_image_mb if kind == "image" else settings.max_video_mb
    data = await _read_limited(file, limit)
    analyzer = get_analyzer()
    name = os.path.basename(file.filename or kind)
    try:
        fn = analyzer.analyze_image if kind == "image" else analyzer.analyze_video
        result = await run_in_threadpool(fn, data, name)
    except InvalidMediaError as exc:
        raise HTTPException(422, str(exc)) from exc
    results.put(result)
    return result


@router.get("/report/{analysis_id}", summary="Download the PDF forensic report for an analysis")
def report(analysis_id: str) -> Response:
    result = results.get(analysis_id)
    if result is None:
        raise HTTPException(404, "Analysis not found or expired. Run the analysis again.")
    pdf = build_pdf(result)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="deepshield-report-{analysis_id}.pdf"'},
    )
