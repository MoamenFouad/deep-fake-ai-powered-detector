import io
import threading
import time

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.analyzer import FRAME_MAX_SIDE, Analyzer, _shrink
from app.services.classifier import DeepfakeClassifier, FusionHead, _fake_index, build_features
from app.services.face_detector import Face, crop_face
from app.services.imaging import InvalidMediaError, decode_image, fit, overlay_heatmap


@pytest.mark.parametrize(
    "labels, expected",
    [
        ({0: "Realism", 1: "Deepfake"}, 1),
        ({0: "Real", 1: "Fake"}, 1),
        ({0: "Fake", 1: "Real"}, 0),
        ({"0": "real", "1": "fake"}, 1),
        ({0: "ai", 1: "hum"}, 0),
        ({0: "artificial", 1: "human"}, 0),
        ({0: "human", 1: "AI-generated"}, 1),
    ],
)
def test_fake_index(labels, expected):
    assert _fake_index(labels) == expected


def test_fake_index_unknown_labels():
    with pytest.raises(ValueError):
        _fake_index({0: "cat", 1: "dog"})


def test_verdict_bands():
    a = Analyzer(Settings(), detector=None, classifier=None)
    assert a.verdict(0.95) == "fake"
    assert a.verdict(0.05) == "real"
    assert a.verdict(0.52) == "uncertain"
    assert a.confidence(0.5) == 0
    assert a.confidence(1.0) == 1
    assert a.confidence(0.0) == 1


def test_build_features_shape():
    embs = [np.zeros((3, 8)), np.ones((3, 4))]
    probs = [np.full(3, 0.9), np.full(3, 0.1)]
    feats = build_features(embs, probs)
    assert feats.shape == (3, 8 + 4 + 2)
    assert feats[0, -2] > 0 > feats[0, -1]  # logits keep the sign of p - 0.5


def test_crop_face_stays_in_bounds():
    img = np.zeros((100, 100, 3), np.uint8)
    crop = crop_face(img, Face(x=80, y=80, w=30, h=30, score=1.0), margin=0.5)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_overlay_heatmap_matches_input_size():
    img = np.zeros((50, 70, 3), np.uint8)
    out = overlay_heatmap(img, np.random.rand(14, 14))
    assert out.shape == img.shape


def test_decode_image_rejects_garbage():
    with pytest.raises(InvalidMediaError):
        decode_image(b"\x00\x01garbage")


def test_build_features_skips_logits_of_encoders():
    """A plain encoder has no fake probability (NaN): it adds its embedding but no logit column."""
    embs = [np.zeros((3, 8)), np.ones((3, 4))]
    probs = [np.full(3, 0.9), np.full(3, np.nan)]
    feats = build_features(embs, probs)
    assert feats.shape == (3, 8 + 4 + 1)
    assert np.isfinite(feats).all()


def test_fusion_head_applies_calibration(tmp_path):
    base = dict(mean=np.zeros(2, np.float32), std=np.ones(2, np.float32), coef=np.array([2.0, 0.0], np.float32),
                intercept=np.float32(0.0), model_ids=np.array(["a/one"]))
    np.savez(tmp_path / "plain.npz", **base)
    np.savez(tmp_path / "calibrated.npz", **base, cal_a=np.float32(0.5), cal_b=np.float32(0.0))
    x = np.array([[1.0, 0.0]])
    plain = FusionHead(tmp_path / "plain.npz").predict(x)[0]
    calibrated = FusionHead(tmp_path / "calibrated.npz").predict(x)[0]
    assert plain == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert calibrated == pytest.approx(1 / (1 + np.exp(-1.0)))  # same side, less extreme


def test_only_one_request_runs_the_models_at_a_time():
    """Model hooks are shared, so concurrent requests must never be inside the models together."""

    class SlowBackbone:
        short_name, is_classifier = "slow", True
        active, overlapped = 0, False

        def forward(self, images):
            SlowBackbone.active += 1
            SlowBackbone.overlapped |= SlowBackbone.active > 1
            time.sleep(0.02)
            SlowBackbone.active -= 1
            return np.full(len(images), 0.7), np.zeros((len(images), 4))

    clf = DeepfakeClassifier.__new__(DeepfakeClassifier)
    clf.backbones, clf.head, clf._lock = [SlowBackbone()], None, threading.Lock()
    image = Image.new("RGB", (32, 32))
    threads = [threading.Thread(target=clf.score, args=([image],), kwargs={"explain": False}) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not SlowBackbone.overlapped


def test_decode_image_scales_16_bit_pixels():
    buf = io.BytesIO()
    Image.fromarray((np.linspace(0, 65535, 64 * 64).reshape(64, 64)).astype(np.uint16)).save(buf, "PNG")
    img = decode_image(buf.getvalue())
    assert img.shape == (64, 64, 3)
    assert img.min() < 10 and img.max() > 245  # the full range survives, nothing is clipped to white


def test_decode_image_honours_exif_rotation():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    buf = io.BytesIO()
    Image.new("RGB", (80, 40), "red").save(buf, "JPEG", exif=exif)
    assert decode_image(buf.getvalue()).shape[:2] == (80, 40)


def test_fit_never_enlarges():
    small = np.zeros((40, 60, 3), np.uint8)
    assert fit(small, 100) is small
    assert fit(np.zeros((400, 200, 3), np.uint8), 100).shape[:2] == (100, 50)


def test_video_frames_are_bounded():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    assert max(_shrink(frame).shape[:2]) == FRAME_MAX_SIDE


def test_aggregate_video_weights_the_most_suspicious_quarter():
    from app.services.analyzer import aggregate_video

    assert aggregate_video([0.2] * 8) == pytest.approx(0.2)
    # Two suspicious frames out of eight pull the score up more than a plain mean would.
    p = [0.1] * 6 + [0.9, 0.9]
    assert aggregate_video(p) == pytest.approx(0.5 * 0.3 + 0.5 * 0.9)
    assert aggregate_video([0.7]) == pytest.approx(0.7)


def test_fusion_head_reads_video_calibration(tmp_path):
    from app.services.classifier import FusionHead

    base = dict(mean=np.zeros(3, np.float32), std=np.ones(3, np.float32), coef=np.ones(3, np.float32),
                intercept=np.float32(0), model_ids=np.array(["a/x", "b/y"]))
    np.savez(tmp_path / "photo.npz", **base)
    np.savez(tmp_path / "video.npz", **base, cal_a=np.float32(0.5), cal_b=np.float32(0.1),
             video_a=np.float32(1.2), video_b=np.float32(-0.2))
    photo, video = FusionHead(tmp_path / "photo.npz"), FusionHead(tmp_path / "video.npz")
    assert (photo.video_a, photo.video_b) == (1.0, 0.0)
    assert (video.video_a, video.video_b) == pytest.approx((1.2, -0.2))
    x = np.array([[1.0, 1.0, 1.0]])
    assert video.predict(x)[0] == pytest.approx(1 / (1 + np.exp(-(0.5 * 3 + 0.1))))


def test_sideways_photo_is_turned_upright():
    import cv2

    from app.services.analyzer import Analyzer
    from app.services.classifier import StubClassifier

    class UprightOnly:
        """Finds a face only when the image is taller than wide, like a portrait."""

        def detect(self, image, max_faces=None):
            h, w = image.shape[:2]
            return [Face(10, 10, 40, 40, 0.99)] if h > w else []

    portrait = np.random.default_rng(0).integers(0, 255, (120, 80, 3), dtype=np.uint8)
    sideways = cv2.rotate(portrait, cv2.ROTATE_90_CLOCKWISE)
    ok, buf = cv2.imencode(".png", sideways)
    result = Analyzer(Settings(), UprightOnly(), StubClassifier()).analyze_image(buf.tobytes(), "x.png")
    assert "rotated" in result.notes and "no_face" not in result.notes
    assert result.faces[0].box is None
