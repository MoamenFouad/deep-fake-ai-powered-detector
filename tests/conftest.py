import os

# Tests run against a deterministic stub so CI never downloads model weights.
os.environ.setdefault("DS_STUB_MODELS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _face_like_image(size: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 200, np.uint8)
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 4, size // 3), 0, 0, 360, (140, 170, 210), -1)
    return img


@pytest.fixture
def png_bytes() -> bytes:
    ok, buf = cv2.imencode(".png", _face_like_image())
    assert ok
    return buf.tobytes()


@pytest.fixture
def mp4_bytes(tmp_path) -> bytes:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 160))
    for i in range(30):
        writer.write(cv2.resize(_face_like_image(seed=i), (160, 160)))
    writer.release()
    return path.read_bytes()
