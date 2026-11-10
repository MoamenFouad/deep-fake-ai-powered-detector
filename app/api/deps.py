"""Lazily-built singletons shared by the API."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from app.config import settings
from app.schemas import AnalysisResult
from app.services.analyzer import Analyzer

log = logging.getLogger(__name__)
_lock = threading.Lock()
_analyzer: Analyzer | None = None


def get_analyzer() -> Analyzer:
    global _analyzer
    if _analyzer is None:
        with _lock:
            if _analyzer is None:
                from app.services.face_detector import FaceDetector

                if settings.use_stub_models:
                    from app.services.classifier import StubClassifier

                    classifier = StubClassifier()
                else:
                    from app.services.classifier import DeepfakeClassifier

                    classifier = DeepfakeClassifier(settings.model_ids, settings.head_path)
                similar = None
                if not settings.use_stub_models:
                    from app.services.similar import load_index

                    similar = load_index(settings.similar_index_path, settings.model_ids)
                _analyzer = Analyzer(settings, FaceDetector(settings.face_model_path), classifier, similar)
                log.info("Analyzer ready (engine=%s, faces=%s)", classifier.engine, _analyzer.detector.backend)
    return _analyzer


def is_loaded() -> bool:
    return _analyzer is not None


class ResultCache:
    """Small in-memory LRU so a finished analysis can be exported as a PDF."""

    def __init__(self, size: int):
        self.size = size
        self._data: OrderedDict[str, AnalysisResult] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, result: AnalysisResult) -> None:
        with self._lock:
            self._data[result.id] = result
            self._data.move_to_end(result.id)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def get(self, key: str) -> AnalysisResult | None:
        with self._lock:
            return self._data.get(key)


results = ResultCache(settings.report_cache_size)
