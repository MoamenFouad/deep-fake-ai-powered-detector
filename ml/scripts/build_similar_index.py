"""Build the "closest known cases" index (retrieval, the R in RAG).

For every training face crop we already cached the two backbones' embeddings in
data/features/. This script turns them into a small search index:

  features (1,154) -> standardize with the fusion head's statistics
                   -> weight each dimension by |fusion coefficient|
                   -> PCA to 64 dimensions (fit on train only)
                   -> L2-normalize, store as float16

At inference the app projects a new face the same way and returns its 5 nearest
training faces by cosine similarity, with their label and the kind of fake.
Only numbers and labels are stored, never dataset images.

It also sets the "unfamiliar image" threshold. For every training face we find its
closest training face from a different person or video (so a face is never matched
with itself or its own clip) and take the 2nd percentile of those similarities. A new
face whose closest known case is below that value is unlike almost everything the
model was trained on, and the app reports it as inconclusive instead of trusting the
score. The threshold uses training data only; the test split is used just to report
how often it fires and how accurate those faces are.

The script also reports how well a plain 5-nearest-neighbour vote does on the
held-out test split, so the evidence shown in the app is backed by a number.

Usage:
  python ml/scripts/build_similar_index.py --data data/crops
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.classifier import FusionHead, build_features  # noqa: E402
from ml.scripts.train_fusion import metrics  # noqa: E402

KINDS = [
    "ffhq", "stylegan",
    "imdb_wiki", "sd_text2img", "sd_inpainting", "insightface",
    "celebdf_real", "celebdf_swap",
    "dfd_real", "dfd_swap",
]


def kind_of(row: pd.Series) -> str:
    fake = int(row.label) == 1
    if row.source == "ffhq_stylegan":
        return "stylegan" if fake else "ffhq"
    if row.source == "dff":
        if not fake:
            return "imdb_wiki"
        name = Path(row.path).name
        for prefix, kind in (("text2img", "sd_text2img"), ("inpainting", "sd_inpainting"), ("insight", "insightface")):
            if name.startswith(prefix):
                return kind
        return "sd_text2img"
    if row.source == "celebdf":
        return "celebdf_swap" if fake else "celebdf_real"
    return "dfd_swap" if fake else "dfd_real"


def load_cache(short: str, paths: pd.Series, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(cache_dir / f"{short}.npz", allow_pickle=True)
    pos = {str(p): i for i, p in enumerate(z["paths"])}
    idx = np.array([pos[p] for p in paths])
    return z["p"][idx], z["emb"][idx]


def project(features: np.ndarray, head: FusionHead, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    z = (features - head.mean) / head.std * np.abs(head.coef)
    v = (z - mean) @ comps.T
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crops")
    ap.add_argument("--head", default="models/fusion_head.npz")
    ap.add_argument("--out", default="models/similar_index.npz")
    ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--unfamiliar-pct", type=float, default=2.0)
    args = ap.parse_args()

    head = FusionHead(ROOT / args.head)
    df = pd.read_csv(ROOT / args.data / "manifest.csv")
    shorts = [m.split("/")[-1] for m in head.model_ids]
    probs, embs = zip(*(load_cache(s, df["path"], ROOT / "data" / "features") for s in shorts))
    feats = build_features(list(embs), list(probs))

    train = (df.split == "train").to_numpy()
    test = ~train
    zt = (feats[train] - head.mean) / head.std * np.abs(head.coef)
    mean = zt.mean(0)
    _, _, vt = np.linalg.svd(zt - mean, full_matrices=False)
    comps = vt[: args.dims].astype(np.float32)

    vec_train = project(feats[train], head, mean, comps)
    vec_test = project(feats[test], head, mean, comps)
    y_train = df.label.to_numpy()[train]
    y_test = df.label.to_numpy()[test]

    sims = vec_test @ vec_train.T
    top = np.argsort(-sims, axis=1)[:, : args.k]
    knn_p = y_train[top].mean(1)
    test_df = df[test].reset_index(drop=True)
    report = {"k": args.k, "dims": args.dims, "train_size": int(train.sum()),
              "overall": metrics(y_test, knn_p)}
    for src, sub in test_df.groupby("source"):
        report[src] = metrics(sub.label.to_numpy(), knn_p[sub.index.to_numpy()])
    agree = float(((knn_p >= 0.5) == (y_test == 1)).mean())
    report["majority_vote_accuracy"] = round(agree, 4)

    # Unfamiliar-image threshold, from training faces only (leave-group-out).
    groups = df.group.astype(str).to_numpy()[train]
    train_sims = vec_train @ vec_train.T
    train_sims[groups[:, None] == groups[None, :]] = -1.0
    threshold = float(np.percentile(train_sims.max(1), args.unfamiliar_pct))
    flagged = sims.max(1) < threshold
    fused = head.predict(feats[test]) >= 0.5
    correct = fused == (y_test == 1)
    report["unfamiliar"] = {
        "threshold": round(threshold, 4),
        "train_percentile": args.unfamiliar_pct,
        "test_flagged": int(flagged.sum()),
        "test_flagged_share": round(float(flagged.mean()), 4),
        "accuracy_flagged": round(float(correct[flagged].mean()), 4) if flagged.any() else None,
        "accuracy_rest": round(float(correct[~flagged].mean()), 4),
    }
    print(json.dumps(report, indent=1))

    kinds = np.array([KINDS.index(kind_of(r)) for _, r in df[train].iterrows()], dtype=np.int8)
    np.savez_compressed(
        ROOT / args.out,
        shift=head.mean.astype(np.float32), scale=(np.abs(head.coef) / head.std).astype(np.float32),
        mean=mean.astype(np.float32), components=comps,
        vectors=vec_train.astype(np.float16), labels=y_train.astype(np.int8),
        kinds=kinds, kind_names=np.array(KINDS), model_ids=np.array(head.model_ids),
        unfamiliar_threshold=np.float32(threshold),
    )
    (ROOT / "reports" / "similar_index.json").write_text(json.dumps(report, indent=2))
    size = (ROOT / args.out).stat().st_size / 1024
    print(f"Saved {args.out} ({size:.0f} KB, {train.sum()} faces) and reports/similar_index.json")


if __name__ == "__main__":
    main()
