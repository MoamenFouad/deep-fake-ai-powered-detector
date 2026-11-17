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
from app.schemas import AnalysisResult, Box, FaceResult, ProvenanceInfo, SimilarCase, TimelinePoint
from app.services.face_detector import FaceDetector, crop_face
from app.services.imaging import InvalidMediaError, bgr_to_pil, decode_image, fit, overlay_heatmap, to_png_b64
from app.services.provenance import inspect as inspect_provenance

log = logging.getLogger(__name__)

# Sampled video frames are kept at most this wide or tall: 24 frames of 4K video would
# otherwise hold over half a gigabyte. Faces stay far larger than the models need.
FRAME_MAX_SIDE = 1280
# Face crops are scaled down to this before scoring; the models resize to 224-440 px anyway.
CROP_MAX_SIDE = 768


# Longest side of the picture given to the whole-image head (as in training).
WHOLE_MAX_SIDE = 1024
# Unfamiliar faces are still reported as fake above this probability (see reports/wild_metrics.json).
UNFAMILIAR_TRUSTED_FAKE = 0.9


def _shrink(frame: np.ndarray) -> np.ndarray:
    return fit(frame, FRAME_MAX_SIDE)


def aggregate_video(probs) -> float:
    """Whole-video score from per-frame scores: the mean blended with the mean of the most
    suspicious quarter, since a manipulation can be confined to part of a clip."""
    p = np.asarray(probs, dtype=float)
    k = max(1, len(p) // 4)
    return float(0.5 * p.mean() + 0.5 * np.sort(p)[-k:].mean())


class Analyzer:
    def __init__(self, settings: Settings, detector: FaceDetector, classifier, similar=None):
        self.s = settings
        self.detector = detector
        self.classifier = classifier
        self.similar = similar  # SimilarIndex or None

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

    def _similar(self, score) -> list[SimilarCase]:
        if self.similar is None or getattr(score, "features", None) is None:
            return []
        neighbors = self.similar.nearest(score.features, k=self.s.similar_k)[0]
        return [SimilarCase(label="fake" if n.label else "real", kind=n.kind, similarity=round(n.similarity, 4))
                for n in neighbors]

    def _face_result(self, crop, score, box) -> FaceResult:
        crop = fit(crop, 384)  # display size; also keeps the heatmap blur cheap
        heat_png = None
        if score.heatmap is not None:
            heat_png = to_png_b64(overlay_heatmap(crop, score.heatmap))
        similar = self._similar(score)
        # Unfamiliar: even the closest training face is a weak match, so the model is
        # outside what it learned and its score should not be trusted on its own.
        unfamiliar = bool(similar) and self.similar.is_unfamiliar(similar[0].similarity)
        return FaceResult(
            box=box,
            fake_probability=round(score.fake_probability, 4),
            verdict="uncertain" if unfamiliar else self.verdict(score.fake_probability),
            model_scores={k: round(v, 4) for k, v in score.model_scores.items()},
            crop_png=to_png_b64(crop),
            heatmap_png=heat_png,
            explains=getattr(score, "explains", "fake"),
            similar=similar,
            unfamiliar=unfamiliar,
        )

    # -------------------------------------------------------------------- image
    def analyze_image(self, data: bytes, filename: str) -> AnalysisResult:
        t0 = time.perf_counter()
        image = decode_image(data)
        notes: list[str] = []
        prov = inspect_provenance(data)
        faces = self.detector.detect(image)
        rotated = None
        if not faces:
            # The face detector only finds upright faces. A photo saved sideways without
            # orientation metadata is retried at 90, 180 and 270 degrees.
            for turn in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
                turned = cv2.rotate(image, turn)
                faces = self.detector.detect(turned)
                if faces:
                    rotated = turned
                    notes.append("rotated")
                    break
        if faces:
            source = image if rotated is None else rotated
            crops = [fit(crop_face(source, f, self.s.face_margin), CROP_MAX_SIDE) for f in faces]
            # Boxes are only drawn when they match the uploaded orientation.
            boxes = [Box(x=f.x, y=f.y, w=f.w, h=f.h) if rotated is None else None for f in faces]
        else:
            crops, boxes = [fit(image, CROP_MAX_SIDE)], [None]
            notes.append("no_face")

        scores = self.classifier.score([bgr_to_pil(c) for c in crops], explain=True)
        face_results = [self._face_result(c, s, b) for c, s, b in zip(crops, scores, boxes)]
        # An image is manipulated if any face in it is manipulated.
        decisive = max(face_results, key=lambda f: f.fake_probability)
        # The whole picture is a second witness: scenes, lighting and rendering give away
        # generators whose faces alone look clean.
        p_whole = None
        if hasattr(self.classifier, "score_whole"):
            p_whole = self.classifier.score_whole(bgr_to_pil(fit(source if faces else image, WHOLE_MAX_SIDE)))
        p = decisive.fake_probability
        if faces and p_whole is not None:
            p = self.classifier.combine(p, p_whole)
        if len(face_results) > 1:
            notes.append("multiple_faces")
        verdict, confidence = self.verdict(p), self.confidence(p)
        if not faces:
            # The models were trained on faces only, so a score for a faceless image is not a verdict.
            verdict, confidence = "uncertain", 0.0
        elif decisive.unfamiliar and p < UNFAMILIAR_TRUSTED_FAKE:
            # An unfamiliar face is not trusted to be real, but a very confident fake call is:
            # on the 2026 test images every unfamiliar face scored 0.9 or more was a fake.
            verdict, confidence = "uncertain", 0.0
            notes.append("unfamiliar")
        if prov.ai:
            # The file says itself that AI made it. A pixel score can be fooled; this label cannot be
            # added by accident, so it decides the verdict.
            p = max(p, 0.99)
            verdict, confidence = "fake", 1.0
            notes.append("provenance_ai")

        return AnalysisResult(
            id=uuid.uuid4().hex[:12],
            media_type="image",
            filename=filename,
            width=image.shape[1],
            height=image.shape[0],
            verdict=verdict,
            fake_probability=round(p, 4),
            confidence=confidence,
            faces=face_results,
            frames_analyzed=1,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            engine=self.classifier.engine,
            notes=notes,
            whole_image_probability=None if p_whole is None else round(p_whole, 4),
            provenance=ProvenanceInfo(**{k: getattr(prov, k) for k in ProvenanceInfo.model_fields}),
        )

    # -------------------------------------------------------------------- video
    def _sample_frames(self, path: str) -> tuple[list[tuple[int, float, np.ndarray]], float | None]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise InvalidMediaError("The file could not be read as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not 1.0 <= fps <= 240.0:  # broken metadata
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[tuple[int, float, np.ndarray]] = []

        if total > 0:
            max_frame = min(total, int(self.s.video_max_seconds * fps))
            n = min(self.s.video_max_frames, max_frame)
            for idx in np.linspace(0, max_frame - 1, n).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    frames.append((int(idx), idx / fps, _shrink(frame)))
        if not frames:
            # Containers without a reliable frame count or without seeking: count the
            # frames first (grab() does not decode), then decode only the ones we keep.
            cap.release()
            cap = cv2.VideoCapture(path)
            limit = int(self.s.video_max_seconds * fps)
            total = 0
            while total < limit and cap.grab():
                total += 1
            cap.release()
            cap = cv2.VideoCapture(path)
            picks = set(np.linspace(0, total - 1, min(self.s.video_max_frames, total)).astype(int).tolist()) \
                if total else set()
            for idx in range(total):
                if not cap.grab():
                    break
                if idx in picks:
                    ok, frame = cap.retrieve()
                    if ok:
                        frames.append((idx, idx / fps, _shrink(frame)))
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
                crops.append(fit(crop_face(frame, faces[0], self.s.face_margin), CROP_MAX_SIDE))
                owners.append((len(timeline) - 1, Box(x=faces[0].x, y=faces[0].y, w=faces[0].w, h=faces[0].h)))

        if not crops:  # no faces anywhere: fall back to whole frames
            notes.append("no_face")
            for i, (_, _, frame) in enumerate(frames):
                crops.append(fit(frame, CROP_MAX_SIDE))
                owners.append((i, None))

        scores = self.classifier.score([bgr_to_pil(c) for c in crops], explain=False, media="video")
        probs = np.array([s.fake_probability for s in scores])
        for (ti, _), p in zip(owners, probs):
            timeline[ti].fake_probability = round(float(p), 4)

        p_video = self.classifier.calibrate_video(aggregate_video(probs))
        if len(crops) < len(frames) and "no_face" not in notes:
            notes.append("partial_faces")

        top = int(np.argmax(probs))
        explained = self.classifier.score([bgr_to_pil(crops[top])], explain=True, media="video")[0]
        explained.fake_probability = float(probs[top])
        top_face = self._face_result(crops[top], explained, owners[top][1])
        if top_face.unfamiliar:
            notes.append("unfamiliar_frame")
        verdict, confidence = self.verdict(p_video), self.confidence(p_video)
        if "no_face" in notes:
            verdict, confidence = "uncertain", 0.0

        return AnalysisResult(
            id=uuid.uuid4().hex[:12],
            media_type="video",
            filename=filename,
            width=frames[0][2].shape[1],
            height=frames[0][2].shape[0],
            verdict=verdict,
            fake_probability=round(p_video, 4),
            confidence=confidence,
            faces=[top_face],
            timeline=timeline,
            frames_analyzed=len(frames),
            duration_seconds=round(duration, 2) if duration else None,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            engine=getattr(self.classifier, "video_engine", self.classifier.engine),
            notes=notes,
        )
