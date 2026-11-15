"""Image helpers: decoding, encoding and heatmap overlays."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

# Photos larger than this are refused: a small file can expand to gigabytes in memory.
MAX_PIXELS = 60_000_000
# Larger photos are scaled down to this longest side; faces stay far bigger than the models need.
MAX_SIDE = 4096


class InvalidMediaError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes into a BGR array, honouring EXIF orientation."""
    try:
        pil = Image.open(io.BytesIO(data))
        width, height = pil.size
    except Exception as exc:
        raise InvalidMediaError("The file could not be read as an image.") from exc
    if width * height > MAX_PIXELS:
        raise InvalidMediaError(
            f"The image is too large ({width} x {height} pixels). The limit is {MAX_PIXELS // 1_000_000} megapixels.")
    if width < 16 or height < 16:
        raise InvalidMediaError("The image is too small to analyze.")
    try:
        pil.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG only: decode at reduced size when it is huge
        pil = ImageOps.exif_transpose(pil)
        if pil.mode.startswith(("I", "F")):  # 16-bit or float pixels: scale to 8-bit instead of clipping
            arr = np.asarray(pil, dtype=np.float32)
            span = float(arr.max() - arr.min()) or 1.0
            pil = Image.fromarray(((arr - arr.min()) / span * 255).astype(np.uint8))
        pil = pil.convert("RGB")
    except Exception as exc:
        raise InvalidMediaError("The file could not be read as an image.") from exc
    if max(pil.size) > MAX_SIDE:
        pil.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def fit(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Scale an image down so its longest side is at most `max_side` (never up)."""
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image_bgr
    return cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def to_png_b64(image_bgr: np.ndarray, max_side: int = 384) -> str:
    ok, buf = cv2.imencode(".png", fit(image_bgr, max_side))
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
