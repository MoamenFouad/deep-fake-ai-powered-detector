"""Response models returned by the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["fake", "real", "uncertain"]


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceResult(BaseModel):
    box: Box | None = Field(None, description="Face location; null when the whole image was analyzed")
    fake_probability: float = Field(..., ge=0, le=1)
    verdict: Verdict
    model_scores: dict[str, float] = Field(default_factory=dict, description="Per-backbone fake probability")
    crop_png: str | None = Field(None, description="Base64 PNG of the face crop")
    heatmap_png: str | None = Field(None, description="Base64 PNG Grad-CAM overlay")


class TimelinePoint(BaseModel):
    t: float = Field(..., description="Timestamp in seconds")
    frame: int
    fake_probability: float | None = Field(None, description="Null when no face was found in the frame")


class AnalysisResult(BaseModel):
    id: str
    media_type: Literal["image", "video"]
    filename: str
    verdict: Verdict
    fake_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1, description="Distance from the decision boundary, scaled to 0..1")
    faces: list[FaceResult] = Field(default_factory=list)
    timeline: list[TimelinePoint] = Field(default_factory=list)
    frames_analyzed: int = 0
    duration_seconds: float | None = None
    processing_ms: int
    engine: str = Field(..., description="Which scoring engine produced the result")
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    models_loaded: bool
    engine: str
    backbones: list[str]
