"""Closest known cases: nearest-neighbour retrieval over the training faces.

The index (models/similar_index.npz, built by ml/scripts/build_similar_index.py)
holds a 64-dimensional fingerprint of every training face, its real/fake label
and the kind of fake it is. No images are stored. For a new face we project its
fusion-head features the same way and return the most similar known faces, which
the UI shows as supporting evidence next to the verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Neighbor:
    label: int  # 1 = fake, 0 = real
    kind: str
    similarity: float


class SimilarIndex:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.shift = data["shift"]
        self.scale = data["scale"]
        self.mean = data["mean"]
        self.components = data["components"]
        self.vectors = data["vectors"].astype(np.float32)
        self.labels = data["labels"].astype(int)
        self.kinds = data["kinds"].astype(int)
        self.kind_names = [str(k) for k in data["kind_names"]]
        self.model_ids = [str(m) for m in data["model_ids"]]
        log.info("Similar-case index: %d faces", len(self.labels))

    def project(self, features: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(features) - self.shift) * self.scale
        v = (z - self.mean) @ self.components.T
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    def nearest(self, features: np.ndarray, k: int = 5) -> list[list[Neighbor]]:
        sims = self.project(features) @ self.vectors.T
        out = []
        for row in sims:
            top = np.argsort(-row)[:k]
            out.append([Neighbor(int(self.labels[i]), self.kind_names[self.kinds[i]], float(row[i])) for i in top])
        return out


def load_index(path: Path, model_ids: list[str]) -> SimilarIndex | None:
    """Return the index, or None when it is missing or was built for other backbones."""
    if not path.exists():
        return None
    try:
        index = SimilarIndex(path)
    except (KeyError, ValueError, OSError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    if index.model_ids != list(model_ids):
        log.warning("Similar-case index was built for %s, not %s; ignoring it", index.model_ids, model_ids)
        return None
    return index
