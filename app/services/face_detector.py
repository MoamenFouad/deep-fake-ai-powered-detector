"""Face localisation.

Primary detector is OpenCV's YuNet (a tiny ~230 KB CNN that runs fast on CPU).
If its weights are unavailable we fall back to OpenCV's bundled Haar cascade so
the service still works offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

YUNET_REPO = "opencv/face_detection_yunet"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def area(self) -> int:
        return self.w * self.h


def ensure_yunet(path: Path) -> Path | None:
    """Return a local YuNet path, downloading it from the Hugging Face Hub if needed."""
    if path.exists():
        return path
    try:
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(YUNET_REPO, YUNET_FILE, local_dir=str(path.parent))
        return Path(downloaded)
    except Exception as exc:  # network blocked, hub down, etc.
        log.warning("YuNet unavailable (%s); using Haar cascade fallback", exc)
        return None


class FaceDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.75, min_face: int = 40):
        self.min_face = min_face
        self.score_threshold = score_threshold
        yunet_path = ensure_yunet(model_path)
        self._yunet = None
        if yunet_path is not None:
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(yunet_path), "", (320, 320), score_threshold, 0.3, 5000
                )
            except cv2.error as exc:
                log.warning("Failed to load YuNet (%s); using Haar fallback", exc)
        self._haar = None
        if self._yunet is None:
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    @property
    def backend(self) -> str:
        return "yunet" if self._yunet is not None else "haar"

    def detect(self, image_bgr: np.ndarray, max_faces: int = 5) -> list[Face]:
        h, w = image_bgr.shape[:2]
        # Work on a downscaled copy for speed on very large photos.
        scale = min(1.0, 1280 / max(h, w))
        small = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1 else image_bgr
        faces: list[Face] = []

        if self._yunet is not None:
            sh, sw = small.shape[:2]
            self._yunet.setInputSize((sw, sh))
            _, dets = self._yunet.detect(small)
            if dets is not None:
                for d in dets:
                    x, y, fw, fh = (d[:4] / scale).tolist()
                    faces.append(Face(int(x), int(y), int(fw), int(fh), float(d[-1])))
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
            for x, y, fw, fh in rects:
                faces.append(Face(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale), 0.9))

        faces = [f for f in faces if f.w >= self.min_face and f.h >= self.min_face]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces]


def crop_face(image_bgr: np.ndarray, face: Face, margin: float = 0.25) -> np.ndarray:
    """Square crop around a face with extra context (deepfake artifacts often sit at the blend boundary)."""
    h, w = image_bgr.shape[:2]
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    side = max(face.w, face.h) * (1 + 2 * margin)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return image_bgr[y0:y1, x0:x1].copy()
