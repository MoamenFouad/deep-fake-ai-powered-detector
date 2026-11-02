"""High-level analysis pipeline for images and videos."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid

import cv2
import numpy as np

from app.config import Settings
from app.schemas import AnalysisResult, Box, FaceResult, TimelinePoint
from app.services.face_detector import FaceDetector, crop_face
from app.services.imaging import InvalidMediaError, bgr_to_pil, decode_image, overlay_heatmap, to_png_b64

log = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, settings: Settings, detector: FaceDetector, classifier):
        self.s = settings
        self.detector = detector
        self.classifier = classifier

    # ------------------------------------------------------------------ helpers
    def verdict(self, p: float) -> str:
        if p >= self.s.fake_threshold + self.s.uncertain_margin:
            return "fake"
        if p <= self.s.fake_threshold - self.s.uncertain_margin:
            return "real"
        return "uncertain"

    def confidence(self, p: float) -> float:
        span = max(self.s.fake_threshold, 1 - self.s.fake_threshold)
        return round(min(1.0, abs(p - self.s.fake_threshold) / span), 4)

    def _face_result(self, crop, score, box) -> FaceResult:
        heat_png = None
        if score.heatmap is not None:
            heat_png = to_png_b64(overlay_heatmap(crop, score.heatmap))
        return FaceResult(
            box=box,
            fake_probability=round(score.fake_probability, 4),
            verdict=self.verdict(score.fake_probability),
            model_scores={k: round(v, 4) for k, v in score.model_scores.items()},
            crop_png=to_png_b64(crop),
            heatmap_png=heat_png,
        )

    # -------------------------------------------------------------------- image
    def analyze_image(self, data: bytes, filename: str) -> AnalysisResult:
        t0 = time.perf_counter()
        image = decode_image(data)
        notes: list[str] = []
        faces = self.detector.detect(image)
        if faces:
            crops = [crop_face(image, f, self.s.face_margin) for f in faces]
            boxes = [Box(x=f.x, y=f.y, w=f.w, h=f.h) for f in faces]
        else:
            crops, boxes = [image], [None]
            notes.append("no_face")

        scores = self.classifier.score([bgr_to_pil(c) for c in crops], explain=True)
        face_results = [self._face_result(c, s, b) for c, s, b in zip(crops, scores, boxes)]
        # An image is manipulated if any face in it is manipulated.
        p = max(f.fake_probability for f in face_results)
        if len(face_results) > 1:
            notes.append("multiple_faces")

        return AnalysisResult(
            id=uuid.uuid4().hex[:12],
            media_type="image",
            filename=filename,
            verdict=self.verdict(p),
            fake_probability=round(p, 4),
            confidence=self.confidence(p),
            faces=face_results,
            frames_analyzed=1,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            engine=self.classifier.engine,
            notes=notes,
        )

    # -------------------------------------------------------------------- video
    def _sample_frames(self, path: str) -> tuple[list[tuple[int, float, np.ndarray]], float | None]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise InvalidMediaError("The file could not be read as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[tuple[int, float, np.ndarray]] = []

        if total > 0:
            max_frame = min(total, int(self.s.video_max_seconds * fps))
            n = min(self.s.video_max_frames, max_frame)
            for idx in np.linspace(0, max_frame - 1, n).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    frames.append((int(idx), idx / fps, frame))
        if not frames:  # containers without a reliable frame count: read sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            all_frames, idx = [], 0
            while idx < self.s.video_max_seconds * fps:
                ok, frame = cap.read()
                if not ok:
                    break
                all_frames.append((idx, idx / fps, frame))
                idx += 1
            total = len(all_frames)
            if total:
                picks = np.linspace(0, total - 1, min(self.s.video_max_frames, total)).astype(int)
                frames = [all_frames[i] for i in picks]
        cap.release()
        if not frames:
            raise InvalidMediaError("No frames could be decoded from this video.")
        duration = total / fps if total else None
        return frames, duration

    def analyze_video(self, data: bytes, filename: str) -> AnalysisResult:
        t0 = time.perf_counter()
        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            frames, duration = self._sample_frames(path)
        finally:
            os.unlink(path)  # uploads are never kept on disk

        notes: list[str] = []
        crops, owners, timeline = [], [], []
        for frame_idx, t, frame in frames:
            faces = self.detector.detect(frame, max_faces=1)
            timeline.append(TimelinePoint(t=round(t, 2), frame=frame_idx))
            if faces:
                crops.append(crop_face(frame, faces[0], self.s.face_margin))
                owners.append((len(timeline) - 1, Box(x=faces[0].x, y=faces[0].y, w=faces[0].w, h=faces[0].h)))

        if not crops:  # no faces anywhere: fall back to whole frames
            notes.append("no_face")
            for i, (_, _, frame) in enumerate(frames):
                crops.append(frame)
                owners.append((i, None))

        scores = self.classifier.score([bgr_to_pil(c) for c in crops], explain=False)
        probs = np.array([s.fake_probability for s in scores])
        for (ti, _), p in zip(owners, probs):
            timeline[ti].fake_probability = round(float(p), 4)

        # Robust aggregate: blend the mean with the mean of the most suspicious quarter,
        # since manipulations can be confined to part of a clip.
        k = max(1, len(probs) // 4)
        p_video = float(0.5 * probs.mean() + 0.5 * np.sort(probs)[-k:].mean())
        if len(crops) < len(frames) and "no_face" not in notes:
            notes.append("partial_faces")

        top = int(np.argmax(probs))
        explained = self.classifier.score([bgr_to_pil(crops[top])], explain=True)[0]
        explained.fake_probability = float(probs[top])
        top_face = self._face_result(crops[top], explained, owners[top][1])

        return AnalysisResult(
            id=uuid.uuid4().hex[:12],
            media_type="video",
            filename=filename,
            verdict=self.verdict(p_video),
            fake_probability=round(p_video, 4),
            confidence=self.confidence(p_video),
            faces=[top_face],
            timeline=timeline,
            frames_analyzed=len(frames),
            duration_seconds=round(duration, 2) if duration else None,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            engine=self.classifier.engine,
            notes=notes,
        )
