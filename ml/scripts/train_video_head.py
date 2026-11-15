"""Train the video head: a fusion head used only for frames of uploaded videos.

Why a second head
-----------------
The photo head (train_fusion.py) learns from thousands of clean photos: Pexels,
CelebA-HQ, COCO, Flickr and modern AI portraits. That made it much better on photos,
but it also learned that the look of a video frame (compression, motion blur, lower
resolution) points to "fake". On whole test videos its AUC fell from 0.77 to 0.66,
and real video frames scored 0.52 on average.

The app always knows whether an upload is a photo or a video, so videos get their own
head, trained only on frames of the training videos (Celeb-DF v2 and Google DFD,
including their augmented copies), with the same features and backbones.

Method (test videos are never used for any choice)
---------------------------------------------------
1. Choose C by 5-fold cross-validation grouped by person (a Celeb-DF celebrity or a
   DFD actor never sits in two folds), scored on video-level AUC.
2. Calibrate frames with Platt scaling on the out-of-fold logits.
3. Calibrate whole videos: aggregate the out-of-fold frame probabilities of each
   training video exactly as the app does (mean blended with the top quarter) and fit
   a second Platt scaling on those video scores.

    python ml/scripts/train_video_head.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.analyzer import aggregate_video  # noqa: E402
from app.services.classifier import build_features  # noqa: E402
from ml.scripts.build_similar_index import load_cache  # noqa: E402

VIDEO_SOURCES = ("celebdf", "dfd")
C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _per_video(p: np.ndarray, y: np.ndarray, video: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = pd.DataFrame({"p": p, "y": y, "v": video}).groupby("v").agg(p=("p", aggregate_video), y=("y", "first"))
    return d.p.to_numpy(), d.y.to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crops")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--backbones", default="buildborderless/CommunityForensics-DeepfakeDet-ViT,"
                                           "prithivMLmods/deepfake-detector-model-v1")
    ap.add_argument("--out", default="models/video_head.npz")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    model_ids = args.backbones.split(",")
    df = pd.read_csv(ROOT / args.data / "manifest.csv")
    df = df[df.source.isin(VIDEO_SOURCES) & (df.split == "train")].reset_index(drop=True)
    probs, embs = zip(*(load_cache(m.split("/")[-1], df.path, ROOT / args.features) for m in model_ids))
    x = build_features(list(embs), list(probs))
    y = df.label.to_numpy()
    video = df.video.astype(str).to_numpy()
    person = (df.source + ":" + df.group.astype(str)).to_numpy()
    original = (df.aug == 0).to_numpy()
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    folds = list(GroupKFold(args.folds).split(z, y, person))
    print(f"{len(df)} training frames ({original.sum()} originals) from {len(set(video))} videos, "
          f"{len(set(person))} people")

    def out_of_fold(c: float) -> np.ndarray:
        oof = np.zeros(len(y))
        for a, b in folds:
            oof[b] = LogisticRegression(C=c, max_iter=3000).fit(z[a], y[a]).decision_function(z[b])
        return oof

    scores = {}
    for c in C_GRID:
        oof = out_of_fold(c)
        pv, yv = _per_video(1 / (1 + np.exp(-oof[original])), y[original], video[original])
        scores[c] = round(float(roc_auc_score(yv, pv)), 4)
    best = max(scores, key=scores.get)
    print("Video-level AUC per C (grouped by person):", scores, "->", best)

    oof = out_of_fold(best)
    frame_cal = LogisticRegression(C=1e6).fit(oof[:, None], y)
    fa, fb = float(frame_cal.coef_[0, 0]), float(frame_cal.intercept_[0])
    frame_p = 1 / (1 + np.exp(-(fa * oof + fb)))
    pv, yv = _per_video(frame_p[original], y[original], video[original])
    video_cal = LogisticRegression(C=1e6).fit(_logit(pv)[:, None], yv)
    va, vb = float(video_cal.coef_[0, 0]), float(video_cal.intercept_[0])

    head = LogisticRegression(C=best, max_iter=3000).fit(z, y)
    out = ROOT / args.out
    np.savez(out, mean=mean.astype(np.float32), std=std.astype(np.float32),
             coef=head.coef_[0].astype(np.float32), intercept=np.float32(head.intercept_[0]),
             model_ids=np.array(model_ids), cal_a=np.float32(fa), cal_b=np.float32(fb),
             video_a=np.float32(va), video_b=np.float32(vb))
    report = {"sources": list(VIDEO_SOURCES), "training_frames": int(len(df)), "training_videos": int(len(set(video))),
              "people": int(len(set(person))), "cv_video_auc_per_C": {str(k): v for k, v in scores.items()},
              "C": best, "frame_calibration": {"a": round(fa, 4), "b": round(fb, 4)},
              "video_calibration": {"a": round(va, 4), "b": round(vb, 4)},
              "cv_video_auc": scores[best]}
    (ROOT / "reports" / "video_head.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
