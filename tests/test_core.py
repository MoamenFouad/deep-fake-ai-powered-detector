import numpy as np
import pytest

from app.config import Settings
from app.services.analyzer import Analyzer
from app.services.classifier import _fake_index, build_features
from app.services.face_detector import Face, crop_face
from app.services.imaging import InvalidMediaError, decode_image, overlay_heatmap


@pytest.mark.parametrize(
    "labels, expected",
    [
        ({0: "Realism", 1: "Deepfake"}, 1),
        ({0: "Real", 1: "Fake"}, 1),
        ({0: "Fake", 1: "Real"}, 0),
        ({"0": "real", "1": "fake"}, 1),
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
