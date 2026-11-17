"""Deepfake scoring engine.

Architecture
------------
1. Frozen pretrained "backbones" look at every face crop. A backbone is either a
   deepfake classifier (it gives a fake probability and the embedding it feeds its
   own classification layer) or a general vision encoder such as CLIP (embedding only).
2. A light fine-tuned *fusion head* (logistic regression trained in ml/) takes the
   concatenated embeddings plus each classifier's logit and outputs the final fake
   probability. If the head file is missing we fall back to averaging the classifiers.
3. Grad-CAM on the *fused* score, through every backbone, shows where the evidence is.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

FAKE_KEYWORDS = ("fake", "deepfake", "manipulated", "synthetic", "generated", "artificial", "ai")
REAL_KEYWORDS = ("real", "realism", "authentic", "original", "human", "hum")
CAM_GRID = 24  # side of the grid the per-backbone maps are combined on


def _fake_index(id2label: dict) -> int:
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    for idx, name in labels.items():
        if name in FAKE_KEYWORDS or (any(w in name for w in FAKE_KEYWORDS[:5]) and not name.startswith("real")):
            return idx
    for idx, name in labels.items():
        if any(word in name for word in REAL_KEYWORDS):
            return 1 - idx if len(labels) == 2 else idx
    raise ValueError(f"Cannot find a fake class in labels {labels}")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class FaceScore:
    fake_probability: float
    model_scores: dict[str, float]
    heatmap: np.ndarray | None = None  # HxW float in [0, 1]
    features: np.ndarray | None = None  # fusion-head input, used for closest-case retrieval
    explains: str = "fake"  # which side of the decision the heatmap supports: "fake" or "real"


class Backbone:
    """Wraps one Hugging Face vision model: a deepfake classifier or a plain encoder."""

    def __init__(self, model_id: str, device: str = "cpu"):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForImageClassification

        self.torch = torch
        self.model_id = model_id
        self.short_name = model_id.split("/")[-1]
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        self.is_classifier = any("ForImageClassification" in a for a in (config.architectures or []))

        if self.is_classifier:
            self.model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
            # Single-logit models (sigmoid) output P(fake) directly.
            self.sigmoid_head = self.model.config.num_labels == 1
            self.fake_idx = 0 if self.sigmoid_head else _fake_index(self.model.config.id2label)
            self.head_module = self.model.classifier
        else:
            full = AutoModel.from_pretrained(model_id)
            # CLIP and SigLIP checkpoints carry a text tower we do not need.
            self.model = getattr(full, "vision_model", full).to(device).eval()
        # First LayerNorm of the last transformer block: "layernorm_before" (ViT, Swin),
        # "layer_norm1" (CLIP, SigLIP) or "norm1" (DINOv2).
        for param in self.model.parameters():  # frozen: Grad-CAM only needs gradients of activations
            param.requires_grad_(False)
        norms = [m for n, m in self.model.named_modules()
                 if n.endswith(("layernorm_before", "layer_norm1", "norm1"))]
        self.cam_target = norms[-1] if norms else None
        log.info("Loaded %s (%s)", model_id, "classifier" if self.is_classifier else "encoder")

    def _pixels(self, images: list[Image.Image]):
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)

    def _run(self, pixels):
        """One forward pass. Returns (fake logit per image or None, embedding per image)."""
        if not self.is_classifier:
            out = self.model(pixel_values=pixels)
            emb = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
            return None, emb
        captured: dict[str, object] = {}
        handle = self.head_module.register_forward_pre_hook(lambda _m, args: captured.__setitem__("x", args[0]))
        try:
            logits = self.model(pixel_values=pixels).logits
        finally:
            handle.remove()
        emb = captured["x"]
        if emb.dim() == 3:
            emb = emb[:, 0]
        if self.sigmoid_head:
            fake_logit = logits[:, 0]
        elif logits.shape[1] == 2:
            fake_logit = logits[:, self.fake_idx] - logits[:, 1 - self.fake_idx]
        else:  # more than two classes: log-odds of the fake class against the rest
            logp = logits.log_softmax(-1)[:, self.fake_idx]
            fake_logit = logp - self.torch.log1p(-logp.exp().clamp(max=1 - 1e-6))
        return fake_logit, emb

    def forward(self, images: list[Image.Image], batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Return (fake probabilities [N], embeddings [N, D]). Encoders return NaN probabilities."""
        probs_out, embs_out = [], []
        with self.torch.inference_mode():
            for i in range(0, len(images), batch_size):
                fake_logit, emb = self._run(self._pixels(images[i:i + batch_size]))
                n = emb.shape[0]
                probs_out.append(fake_logit.sigmoid().float().cpu().numpy() if fake_logit is not None
                                 else np.full(n, np.nan, np.float32))
                embs_out.append(emb.float().cpu().numpy())
        return np.concatenate(probs_out), np.concatenate(embs_out)

    def run_with_activations(self, image: Image.Image):
        """Differentiable pass for Grad-CAM. Returns (fake logit or None, embedding, block activations)."""
        store: dict[str, object] = {}

        def hook(_m, _i, out):
            out.retain_grad()
            store["act"] = out

        handle = self.cam_target.register_forward_hook(hook) if self.cam_target is not None else None
        try:
            fake_logit, emb = self._run(self._pixels([image]).requires_grad_(True))
        finally:
            if handle is not None:
                handle.remove()
        return (fake_logit[0] if fake_logit is not None else None), emb[0], store.get("act")

    def cam(self, act) -> np.ndarray | None:
        """Grad-CAM map (signed, unnormalised) from activations whose .grad has been filled."""
        if act is None or act.grad is None:
            return None
        a, g = act[0].detach(), act.grad[0].detach()
        side = int(a.shape[0] ** 0.5)
        if side * side != a.shape[0]:  # drop [CLS] / register tokens placed before the patches
            a, g = a[-side * side:], g[-side * side:]
        weights = g.mean(dim=0)
        return (a * weights).sum(-1).reshape(side, side).cpu().numpy()


class FusionHead:
    """Logistic-regression head trained in ml/scripts/train_fusion.py."""

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.mean = data["mean"]
        self.std = data["std"]
        self.coef = data["coef"]
        self.intercept = float(data["intercept"])
        self.model_ids = [str(m) for m in data["model_ids"]]
        # Optional Platt calibration of the head's logit, fitted on held-out folds.
        self.cal_a = float(data["cal_a"]) if "cal_a" in data.files else 1.0
        self.cal_b = float(data["cal_b"]) if "cal_b" in data.files else 0.0
        # Video heads also carry a calibration for the aggregated whole-video score.
        self.video_a = float(data["video_a"]) if "video_a" in data.files else 1.0
        self.video_b = float(data["video_b"]) if "video_b" in data.files else 0.0
        # The whole-image head also carries the stacker that merges it with the face score:
        # logit(p) = stack[0] * logit(p_face) + stack[1] * logit(p_whole) + stack[2].
        self.stack = [float(v) for v in data["stack"]] if "stack" in data.files else None

    def logit(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean) / self.std
        return self.cal_a * (z @ self.coef + self.intercept) + self.cal_b

    def predict(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logit(features))


def build_features(embeddings: list[np.ndarray], probs: list[np.ndarray]) -> np.ndarray:
    """Fusion-head input: [emb_1, ..., emb_k, logit_i for every backbone that is a classifier].

    Encoders have no fake probability (NaN), so they contribute their embedding only.
    """
    logits = [_logit(p) for p in probs if not np.isnan(p).all()]
    parts = [*embeddings, np.stack(logits, axis=1)] if logits else list(embeddings)
    return np.concatenate(parts, axis=1)


class DeepfakeClassifier:
    def __init__(self, model_ids: list[str], head_path: Path, video_head_path: Path | None = None,
                 whole_head_path: Path | None = None):
        self.backbones = [Backbone(m) for m in model_ids]
        # Forward hooks and gradients live on the shared modules, so only one request
        # may run the models at a time. Inference is CPU-bound anyway.
        self._lock = threading.Lock()
        self.head: FusionHead | None = None
        if head_path.exists():
            head = FusionHead(head_path)
            if head.model_ids == list(model_ids):
                self.head = head
            else:
                log.warning("Fusion head was trained on %s, not %s; ignoring it", head.model_ids, model_ids)
        # Optional second head for video frames; videos fall back to the photo head without it.
        self.video_head: FusionHead | None = None
        if video_head_path is not None and video_head_path.exists() and self.head is not None:
            video_head = FusionHead(video_head_path)
            if video_head.model_ids == list(model_ids):
                self.video_head = video_head
            else:
                log.warning("Video head was trained on %s, not %s; ignoring it", video_head.model_ids, model_ids)
        # Optional head that looks at the whole picture with the same backbones (no extra memory).
        self.whole_head: FusionHead | None = None
        if whole_head_path is not None and whole_head_path.exists():
            whole_head = FusionHead(whole_head_path)
            if whole_head.model_ids == list(model_ids) and whole_head.stack is not None:
                self.whole_head = whole_head
            else:
                log.warning("Whole-image head does not match the backbones; ignoring it")
        if self.head is None and not any(b.is_classifier for b in self.backbones):
            raise ValueError("Without a fusion head at least one backbone must be a deepfake classifier")

    @property
    def engine(self) -> str:
        return "fusion-head" if self.head else "ensemble-mean"

    @property
    def video_engine(self) -> str:
        return "video-head" if self.video_head is not None else self.engine

    @property
    def names(self) -> list[str]:
        return [b.short_name for b in self.backbones]

    def _head_for(self, media: str) -> FusionHead | None:
        return self.video_head if media == "video" and self.video_head is not None else self.head

    def calibrate_video(self, p: float) -> float:
        """Map an aggregated video score through the video head's whole-video calibration."""
        head = self._head_for("video")
        if head is None or head is self.head:
            return p
        return float(_sigmoid(head.video_a * _logit(np.array(p)) + head.video_b))

    def score(self, crops: list[Image.Image], explain: bool = True, media: str = "image") -> list[FaceScore]:
        """Score face crops. `media="video"` uses the video head for frames of a video."""
        if not crops:
            return []
        head = self._head_for(media)
        with self._lock:
            probs, embs = zip(*(b.forward(crops) for b in self.backbones))
            features = build_features(list(embs), list(probs))
            if head is not None:
                final = head.predict(features)
            else:
                final = np.nanmean(np.stack(probs), axis=0)

            results = []
            for i, crop in enumerate(crops):
                scores = {b.short_name: float(p[i]) for b, p in zip(self.backbones, probs) if b.is_classifier}
                heatmap, side = self._explain(crop, head) if explain else (None, "fake")
                results.append(FaceScore(float(final[i]), scores, heatmap, features[i], side))
        return results

    def score_whole(self, image: Image.Image) -> float | None:
        """Fake probability of the whole picture, or None without a whole-image head."""
        if self.whole_head is None:
            return None
        with self._lock:
            probs, embs = zip(*(b.forward([image]) for b in self.backbones))
            features = build_features(list(embs), list(probs))
        return float(self.whole_head.predict(features)[0])

    def combine(self, p_face: float, p_whole: float | None) -> float:
        """Merge the decisive face score with the whole-image score (fitted stacker)."""
        if p_whole is None or self.whole_head is None:
            return p_face
        a, b, c = self.whole_head.stack
        return float(_sigmoid(a * _logit(np.array(p_face)) + b * _logit(np.array(p_whole)) + c))

    def _explain(self, crop: Image.Image, head: FusionHead | None) -> tuple[np.ndarray | None, str]:
        """Grad-CAM of the fused score through every backbone, for the side the model chose."""
        torch = self.backbones[0].torch
        with torch.enable_grad():
            runs = [b.run_with_activations(crop) for b in self.backbones]
            logits = [r[0] for r in runs if r[0] is not None]
            if head is not None:
                parts = [r[1] for r in runs] + ([torch.stack(logits).clamp(-13.8, 13.8)] if logits else [])
                x = torch.cat(parts)
                mean, std, coef = (torch.from_numpy(np.asarray(v, np.float32)) for v in
                                   (head.mean, head.std, head.coef))
                score = head.cal_a * (((x - mean) / std) @ coef + head.intercept) + head.cal_b
            else:
                score = torch.stack(logits).mean()
            side = "fake" if float(score.detach()) >= 0 else "real"
            (score if side == "fake" else -score).backward()

        total = np.zeros((CAM_GRID, CAM_GRID), np.float32)
        for b, run in zip(self.backbones, runs):
            cam = b.cam(run[2])
            if cam is not None:
                # Every map is a first-order share of the same fused score, so they add up.
                t = torch.from_numpy(cam)[None, None]
                total += torch.nn.functional.interpolate(t, size=(CAM_GRID, CAM_GRID), mode="bilinear",
                                                         align_corners=False)[0, 0].numpy()
        total = np.maximum(total, 0)
        if total.max() <= 0:
            return None, side
        return total / total.max(), side


class StubClassifier:
    """Deterministic, dependency-free stand-in used by tests and CI."""

    engine = "stub"
    names = ["stub-a", "stub-b"]

    def calibrate_video(self, p: float) -> float:
        return p

    def score_whole(self, image: Image.Image) -> float | None:
        return None

    def combine(self, p_face: float, p_whole: float | None) -> float:
        return p_face

    def score(self, crops: list[Image.Image], explain: bool = True, media: str = "image") -> list[FaceScore]:
        out = []
        for crop in crops:
            arr = np.asarray(crop.convert("L").resize((32, 32)), dtype=np.float32)
            digest = hashlib.sha256(arr.tobytes()).digest()
            p = digest[0] / 255.0
            heat = None
            if explain:
                yy, xx = np.mgrid[0:14, 0:14]
                heat = np.exp(-((xx - 7) ** 2 + (yy - 6) ** 2) / 18.0)
            out.append(FaceScore(p, {"stub-a": p, "stub-b": 1 - p}, heat, None, "fake" if p >= 0.5 else "real"))
        return out
