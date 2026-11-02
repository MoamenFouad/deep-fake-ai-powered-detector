"""Deepfake scoring engine.

Architecture
------------
1. Two pretrained deepfake classifiers ("backbones", a ViT and a SigLIP) score
   every face crop and expose their pooled embeddings.
2. A light fine-tuned *fusion head* (logistic regression trained in ml/) takes the
   concatenated embeddings plus each backbone's logit and outputs the final fake
   probability. If the head file is missing we fall back to averaging backbones.
3. Grad-CAM on the primary backbone explains *where* the evidence is.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

FAKE_KEYWORDS = ("fake", "deepfake", "manipulated", "synthetic", "generated")
REAL_KEYWORDS = ("real", "realism", "authentic", "original", "human")


def _fake_index(id2label: dict) -> int:
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    for idx, name in labels.items():
        if any(word in name for word in FAKE_KEYWORDS) and not name.startswith("real"):
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


class Backbone:
    """Wraps one Hugging Face image classifier."""

    def __init__(self, model_id: str, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        self.model_id = model_id
        self.short_name = model_id.split("/")[-1]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
        self.device = device
        # Single-logit models (sigmoid) output P(fake) directly.
        self.sigmoid_head = self.model.config.num_labels == 1
        self.fake_idx = 0 if self.sigmoid_head else _fake_index(self.model.config.id2label)
        self.head_module = self.model.classifier
        # Last pre-attention LayerNorm: works for ViT ("layernorm_before") and SigLIP ("layer_norm1").
        norms = [m for n, m in self.model.named_modules()
                 if n.endswith("layernorm_before") or n.endswith("layer_norm1")]
        self.cam_target = norms[-1]
        log.info("Loaded %s (fake index %d, sigmoid=%s)", model_id, self.fake_idx, self.sigmoid_head)

    def _pixels(self, images: list[Image.Image]):
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)

    def _probs(self, logits):
        if self.sigmoid_head:
            return logits[:, 0].sigmoid()
        return logits.softmax(-1)[:, self.fake_idx]

    def forward(self, images: list[Image.Image], batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Return (fake probabilities [N], pooled embeddings [N, D]).

        The embedding is whatever the model feeds its classification head, captured
        with a hook, so this works across ViT, SigLIP and similar architectures.
        """
        probs_out, embs_out = [], []
        captured: dict[str, object] = {}
        handle = self.head_module.register_forward_pre_hook(
            lambda _m, args: captured.__setitem__("x", args[0]))
        try:
            with self.torch.inference_mode():
                for i in range(0, len(images), batch_size):
                    logits = self.model(pixel_values=self._pixels(images[i:i + batch_size])).logits
                    emb = captured["x"]
                    if emb.dim() == 3:
                        emb = emb[:, 0]
                    probs_out.append(self._probs(logits).float().cpu().numpy())
                    embs_out.append(emb.float().cpu().numpy())
        finally:
            handle.remove()
        return np.concatenate(probs_out), np.concatenate(embs_out)

    def gradcam(self, image: Image.Image) -> np.ndarray:
        """Grad-CAM over the patch tokens of the final transformer block."""
        torch = self.torch
        target = self.cam_target
        store: dict[str, torch.Tensor] = {}

        def fwd_hook(_m, _i, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = target.register_forward_hook(fwd_hook)
        try:
            pixels = self._pixels([image])
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                logits = self.model(pixel_values=pixels).logits
                # Gradient of the "fake" evidence (logit difference for 2-class heads).
                if self.sigmoid_head:
                    target_score = logits[0, 0]
                else:
                    target_score = logits[0, self.fake_idx] - logits[0, 1 - self.fake_idx]
                target_score.backward()
        finally:
            handle.remove()

        act, grad = store["act"][0].detach(), store["grad"][0].detach()
        n = act.shape[0]
        if int(round((n - 1) ** 0.5)) ** 2 == n - 1:  # ViT: drop the [CLS] token
            act, grad = act[1:], grad[1:]
        side = int(round(act.shape[0] ** 0.5))
        act, grad = act[: side * side], grad[: side * side]
        weights = grad.mean(dim=0)
        cam = torch.relu((act * weights).sum(-1)).reshape(side, side).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


class FusionHead:
    """Logistic-regression head trained in ml/scripts/train_fusion.py."""

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.mean = data["mean"]
        self.std = data["std"]
        self.coef = data["coef"]
        self.intercept = float(data["intercept"])
        self.model_ids = [str(m) for m in data["model_ids"]]

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean) / self.std
        return _sigmoid(z @ self.coef + self.intercept)


def build_features(embeddings: list[np.ndarray], probs: list[np.ndarray]) -> np.ndarray:
    """Feature vector used by the fusion head: [emb_1, ..., emb_k, logit_1, ..., logit_k]."""
    logits = np.stack([_logit(p) for p in probs], axis=1)
    return np.concatenate([*embeddings, logits], axis=1)


class DeepfakeClassifier:
    def __init__(self, model_ids: list[str], head_path: Path):
        self.backbones = [Backbone(m) for m in model_ids]
        self.head: FusionHead | None = None
        if head_path.exists():
            head = FusionHead(head_path)
            if head.model_ids == list(model_ids):
                self.head = head
            else:
                log.warning("Fusion head was trained on %s, not %s; ignoring it", head.model_ids, model_ids)

    @property
    def engine(self) -> str:
        return "fusion-head" if self.head else "ensemble-mean"

    @property
    def names(self) -> list[str]:
        return [b.short_name for b in self.backbones]

    def score(self, crops: list[Image.Image], explain: bool = True) -> list[FaceScore]:
        if not crops:
            return []
        probs, embs = zip(*(b.forward(crops) for b in self.backbones))
        if self.head is not None:
            final = self.head.predict(build_features(list(embs), list(probs)))
        else:
            final = np.mean(np.stack(probs), axis=0)

        results = []
        for i, crop in enumerate(crops):
            scores = {b.short_name: float(p[i]) for b, p in zip(self.backbones, probs)}
            heatmap = self.backbones[0].gradcam(crop) if explain else None
            results.append(FaceScore(float(final[i]), scores, heatmap))
        return results


class StubClassifier:
    """Deterministic, dependency-free stand-in used by tests and CI."""

    engine = "stub"
    names = ["stub-a", "stub-b"]

    def score(self, crops: list[Image.Image], explain: bool = True) -> list[FaceScore]:
        out = []
        for crop in crops:
            arr = np.asarray(crop.convert("L").resize((32, 32)), dtype=np.float32)
            digest = hashlib.sha256(arr.tobytes()).digest()
            p = digest[0] / 255.0
            heat = None
            if explain:
                yy, xx = np.mgrid[0:14, 0:14]
                heat = np.exp(-((xx - 7) ** 2 + (yy - 6) ** 2) / 18.0)
            out.append(FaceScore(p, {"stub-a": p, "stub-b": 1 - p}, heat))
        return out
