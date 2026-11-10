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


def test_analyze_image(client, png_bytes):
    r = client.post("/api/analyze", files={"file": ("face.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["media_type"] == "image"
    assert body["verdict"] in {"fake", "real", "uncertain"}
    assert 0 <= body["fake_probability"] <= 1
    assert 0 <= body["confidence"] <= 1
    assert body["faces"], "expected at least one analyzed region"
    face = body["faces"][0]
    assert face["crop_png"] and face["heatmap_png"]
    assert set(face["model_scores"]) == {"stub-a", "stub-b"}
    assert face["similar"] == []


def test_analyze_video_returns_timeline(client, mp4_bytes):
    r = client.post("/api/analyze", files={"file": ("clip.mp4", mp4_bytes, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["media_type"] == "video"
    assert body["frames_analyzed"] > 1
    assert len(body["timeline"]) == body["frames_analyzed"]
    assert body["duration_seconds"] and body["duration_seconds"] > 2
    assert len(body["faces"]) == 1


def test_report_pdf(client, png_bytes):
    analysis = client.post("/api/analyze", files={"file": ("face.png", png_bytes, "image/png")}).json()
    r = client.get(f"/api/report/{analysis['id']}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


def test_report_unknown_id(client):
    assert client.get("/api/report/doesnotexist").status_code == 404


def test_rejects_unsupported_type(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/api/analyze", files={"file": ("broken.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422


def test_rejects_empty_file(client):
    r = client.post("/api/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_rejects_oversized_image(client, monkeypatch):
    from dataclasses import replace

    from app.api import routes

    monkeypatch.setattr(routes, "settings", replace(routes.settings, max_image_mb=0.0001))
    r = client.post("/api/analyze", files={"file": ("big.png", b"x" * 1024, "image/png")})
    assert r.status_code == 413
