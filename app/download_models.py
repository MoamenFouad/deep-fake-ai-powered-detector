"""Pre-download every model the service needs (used at Docker build time).

    python -m app.download_models
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

from app.config import settings
from app.services.face_detector import ensure_yunet


def main() -> None:
    for model_id in settings.model_ids:
        print(f"Downloading {model_id}")
        snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"],
                          ignore_patterns=["checkpoint-*/*", "*/optimizer*"])
    path = ensure_yunet(settings.face_model_path)
    print(f"Face detector: {path or 'Haar fallback'}")


if __name__ == "__main__":
    main()
