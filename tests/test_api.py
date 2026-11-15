import re
from pathlib import Path

import cv2
import numpy as np

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


def _png(width: int, height: int) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((height, width, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["similar_cases"] is False  # the stub model has no retrieval index


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "DeepShield" in r.text


def test_page_assets_exist(client):
    """Every script, stylesheet and sample the page points at must be served."""
    html = (STATIC / "index.html").read_text()
    urls = set(re.findall(r'(?:src|href|data-sample)="(/static/[^"]+)"', html))
    assert urls, "the page should reference static files"
    for url in urls:
        assert client.get(url).status_code == 200, url


def test_metrics_served(client):
    body = client.get("/api/metrics").json()
    assert body["test"]["fusion_head"]["overall"]["auc"] > 0.5


def test_analyze_image(client, png_bytes):
    r = client.post("/api/analyze", files={"file": ("face.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["media_type"] == "image"
    assert body["verdict"] in {"fake", "real", "uncertain"}
    assert 0 <= body["fake_probability"] <= 1
    assert 0 <= body["confidence"] <= 1
    assert (body["width"], body["height"]) == (320, 320)
    assert body["notes"] == []
    face = body["faces"][0]
    assert face["box"] == {"x": 80, "y": 80, "w": 160, "h": 160}
    assert face["crop_png"] and face["heatmap_png"]
    assert face["explains"] in {"fake", "real"}
    assert set(face["model_scores"]) == {"stub-a", "stub-b"}
    assert face["similar"] == []


def test_image_without_a_face_is_inconclusive(client, png_bytes, no_faces):
    body = client.post("/api/analyze", files={"file": ("scene.png", png_bytes, "image/png")}).json()
    assert body["verdict"] == "uncertain" and body["confidence"] == 0
    assert "no_face" in body["notes"]
    assert body["faces"][0]["box"] is None


def test_analyze_video_returns_timeline(client, mp4_bytes):
    r = client.post("/api/analyze", files={"file": ("clip.mp4", mp4_bytes, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["media_type"] == "video"
    assert body["frames_analyzed"] > 1
    assert len(body["timeline"]) == body["frames_analyzed"]
    assert all(p["fake_probability"] is not None for p in body["timeline"])
    assert body["duration_seconds"] and body["duration_seconds"] > 2
    assert len(body["faces"]) == 1


def test_video_without_a_face_is_inconclusive(client, mp4_bytes, no_faces):
    body = client.post("/api/analyze", files={"file": ("clip.mp4", mp4_bytes, "video/mp4")}).json()
    assert body["verdict"] == "uncertain" and "no_face" in body["notes"]


def test_report_pdf(client, png_bytes):
    analysis = client.post("/api/analyze", files={"file": ("face.png", png_bytes, "image/png")}).json()
    r = client.get(f"/api/report/{analysis['id']}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


def test_report_pdf_survives_markup_in_the_filename(client, png_bytes):
    name = 'me & <b>friends</b> "2026".png'
    analysis = client.post("/api/analyze", files={"file": (name, png_bytes, "image/png")}).json()
    r = client.get(f"/api/report/{analysis['id']}")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")


def test_report_pdf_for_video(client, mp4_bytes):
    analysis = client.post("/api/analyze", files={"file": ("clip.mp4", mp4_bytes, "video/mp4")}).json()
    r = client.get(f"/api/report/{analysis['id']}")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")


def test_report_unknown_id(client):
    assert client.get("/api/report/doesnotexist").status_code == 404


def test_rejects_unsupported_type(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/api/analyze", files={"file": ("broken.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422


def test_rejects_corrupt_video(client):
    r = client.post("/api/analyze", files={"file": ("broken.mp4", b"not a video" * 50, "video/mp4")})
    assert r.status_code == 422


def test_rejects_empty_file(client):
    r = client.post("/api/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_rejects_oversized_file(client, monkeypatch):
    from dataclasses import replace

    from app.api import routes

    monkeypatch.setattr(routes, "settings", replace(routes.settings, max_image_mb=0.0001))
    r = client.post("/api/analyze", files={"file": ("big.png", b"x" * 1024, "image/png")})
    assert r.status_code == 413


def test_rejects_too_many_pixels(client, monkeypatch):
    from app.services import imaging

    monkeypatch.setattr(imaging, "MAX_PIXELS", 10_000)
    r = client.post("/api/analyze", files={"file": ("wide.png", _png(200, 100), "image/png")})
    assert r.status_code == 422 and "too large" in r.json()["detail"]


def test_rejects_tiny_image(client):
    r = client.post("/api/analyze", files={"file": ("dot.png", _png(8, 8), "image/png")})
    assert r.status_code == 422


def test_very_large_image_is_scaled_down(client, monkeypatch):
    from app.services import imaging

    monkeypatch.setattr(imaging, "MAX_SIDE", 100)
    body = client.post("/api/analyze", files={"file": ("big.png", _png(400, 200), "image/png")}).json()
    assert (body["width"], body["height"]) == (100, 50)


def test_answers_503_while_models_load(client, png_bytes, monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes, "is_loaded", lambda: False)
    r = client.post("/api/analyze", files={"file": ("face.png", png_bytes, "image/png")})
    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    assert client.get("/api/health").json()["models_loaded"] is False
