"""Image helpers: decoding, encoding and heatmap overlays."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from PIL import Image, ImageOps


class InvalidMediaError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes into a BGR array, honouring EXIF orientation."""
    try:
        import io

        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:
        raise InvalidMediaError("The file could not be read as an image.") from exc
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def to_png_b64(image_bgr: np.ndarray, max_side: int = 384) -> str:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    heat = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(1.0, w / 40))
    heat = np.clip(heat / (heat.max() + 1e-8), 0, 1)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
