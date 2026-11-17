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


class SimilarCase(BaseModel):
    label: Literal["real", "fake"]
    kind: str = Field(..., description="Source of the known face, e.g. stylegan, sd_inpainting, celebdf_swap, ffhq")
    similarity: float = Field(..., description="Cosine similarity in the retrieval space, -1..1")


class FaceResult(BaseModel):
    box: Box | None = Field(None, description="Face location; null when the whole image was analyzed")
    fake_probability: float = Field(..., ge=0, le=1)
    verdict: Verdict
    model_scores: dict[str, float] = Field(default_factory=dict, description="Per-backbone fake probability")
    crop_png: str | None = Field(None, description="Base64 PNG of the face crop")
    heatmap_png: str | None = Field(None, description="Base64 PNG Grad-CAM overlay of the fused score")
    explains: Literal["fake", "real"] = Field(
        "fake", description="Which side of the decision the warm areas of the heatmap pushed toward")
    similar: list[SimilarCase] = Field(
        default_factory=list, description="Closest known faces from the training set (labels only, no images)")
    unfamiliar: bool = Field(
        False, description="True when even the closest known face is a weak match, so the score is not trusted")


class TimelinePoint(BaseModel):
    t: float = Field(..., description="Timestamp in seconds")
    frame: int
    fake_probability: float | None = Field(None, description="Null when no face was found in the frame")


class ProvenanceInfo(BaseModel):
    ai: bool = Field(False, description="True when the file's own metadata says AI made or edited it")
    kind: Literal["generated", "edited"] | None = None
    source: Literal["c2pa", "xmp", "exif", "png"] | None = Field(
        None, description="Where the label was found: C2PA Content Credentials, XMP, EXIF or PNG text")
    tool: str | None = Field(None, description="Who made or signed the file, when known")
    signed: bool = Field(False, description="True for C2PA Content Credentials with a valid signature")
    camera: str | None = Field(None, description="Camera make and model from EXIF. Easy to fake, so never proof")


class AnalysisResult(BaseModel):
    id: str
    media_type: Literal["image", "video"]
    filename: str
    width: int | None = Field(None, description="Width in pixels of the analyzed image or frame; boxes use this scale")
    height: int | None = None
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
    provenance: ProvenanceInfo | None = Field(None, description="What the file's own metadata says about its origin")
    whole_image_probability: float | None = Field(
        None, ge=0, le=1, description="Fake probability from the whole-image detector, for photos")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    models_loaded: bool
    engine: str
    backbones: list[str]
    similar_cases: bool = False
