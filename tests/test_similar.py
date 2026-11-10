"""Closest-known-cases retrieval: index maths and how the analyzer reports it."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.config import settings
from app.services.analyzer import Analyzer
from app.services.classifier import FaceScore
from app.services.face_detector import FaceDetector
from app.services.similar import SimilarIndex, load_index

DIM = 6


@pytest.fixture()
def index_path(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(8, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    path = tmp_path / "index.npz"
    np.savez(
        path,
        shift=np.zeros(DIM, np.float32), scale=np.ones(DIM, np.float32),
        mean=np.zeros(DIM, np.float32), components=np.eye(3, DIM, dtype=np.float32),
        vectors=vectors.astype(np.float16), labels=np.array([1, 1, 1, 1, 0, 0, 0, 0], np.int8),
        kinds=np.array([1, 1, 3, 3, 0, 0, 2, 2], np.int8),
        kind_names=np.array(["ffhq", "stylegan", "imdb_wiki", "sd_inpainting"]),
        model_ids=np.array(["a/one", "b/two"]),
    )
    return path, vectors


def test_nearest_returns_itself_first(index_path):
    path, vectors = index_path
    index = SimilarIndex(path)
    query = np.zeros(DIM)
    query[:3] = vectors[5]
    top = index.nearest(query, k=3)[0]
    assert len(top) == 3
    assert top[0].similarity == pytest.approx(1.0, abs=1e-2)
    assert top[0].label == 0 and top[0].kind == "ffhq"
    assert top[0].similarity >= top[1].similarity >= top[2].similarity


def test_load_index_checks_backbones(index_path, tmp_path):
    path, _ = index_path
    assert load_index(path, ["a/one", "b/two"]) is not None
    assert load_index(path, ["other/model"]) is None
    assert load_index(tmp_path / "missing.npz", ["a/one", "b/two"]) is None


class _FeatureStub:
    engine = "stub"
    names = ["a", "b"]

    def score(self, crops, explain=True):
        feats = np.zeros(DIM)
        feats[0] = 1.0
        return [FaceScore(0.9, {"a": 0.9, "b": 0.8}, None, feats) for _ in crops]


def test_analyzer_attaches_similar_cases(index_path):
    path, _ = index_path
    analyzer = Analyzer(replace(settings, similar_k=4), FaceDetector(settings.face_model_path), _FeatureStub(),
                        SimilarIndex(path))
    ok, png = cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))
    result = analyzer.analyze_image(png.tobytes(), "x.png")
    similar = result.faces[0].similar
    assert len(similar) == 4
    assert all(s.label in ("real", "fake") for s in similar)
    assert similar[0].similarity >= similar[-1].similarity


def test_no_index_means_no_similar_cases():
    analyzer = Analyzer(settings, FaceDetector(settings.face_model_path), _FeatureStub(), None)
    ok, png = cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))
    assert analyzer.analyze_image(png.tobytes(), "x.png").faces[0].similar == []
