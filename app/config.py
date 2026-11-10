"""Central configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = "DeepShield"
    version: str = "1.2.0"

    # Pretrained backbones (Hugging Face model ids). Each must be an image
    # classifier whose labels contain a "fake"/"deepfake" class.
    model_ids: list[str] = field(
        default_factory=lambda: _env_list(
            "DS_MODELS",
            "buildborderless/CommunityForensics-DeepfakeDet-ViT,prithivMLmods/deepfake-detector-model-v1",
        )
    )
    # Fine-tuned stacking head trained on top of the backbones (see ml/).
    head_path: Path = Path(os.getenv("DS_HEAD_PATH", ROOT_DIR / "models" / "fusion_head.npz"))
    # Closest-known-cases index built from the training faces (see ml/scripts/build_similar_index.py).
    similar_index_path: Path = Path(os.getenv("DS_SIMILAR_INDEX", ROOT_DIR / "models" / "similar_index.npz"))
    similar_k: int = int(os.getenv("DS_SIMILAR_K", "5"))
    face_model_path: Path = Path(
        os.getenv("DS_FACE_MODEL", ROOT_DIR / "models" / "face_detection_yunet_2023mar.onnx")
    )

    # Use a deterministic stub instead of real models (tests and CI).
    use_stub_models: bool = os.getenv("DS_STUB_MODELS", "0") == "1"

    max_image_mb: float = float(os.getenv("DS_MAX_IMAGE_MB", "10"))
    max_video_mb: float = float(os.getenv("DS_MAX_VIDEO_MB", "60"))
    video_max_frames: int = int(os.getenv("DS_VIDEO_FRAMES", "24"))
    video_max_seconds: float = float(os.getenv("DS_VIDEO_MAX_SECONDS", "120"))

    # Decision thresholds on the fake probability.
    fake_threshold: float = float(os.getenv("DS_FAKE_THRESHOLD", "0.5"))
    uncertain_margin: float = float(os.getenv("DS_UNCERTAIN_MARGIN", "0.1"))

    face_margin: float = 0.25
    report_cache_size: int = 64

    image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp", "image/bmp")
    video_types: tuple[str, ...] = (
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
        "video/x-matroska",
    )


settings = Settings()
