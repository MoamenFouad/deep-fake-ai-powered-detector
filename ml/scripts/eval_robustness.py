"""How much does the verdict change when a held-out face is degraded?

A sample of held-out crops (test and unseen splits) is scored clean and again after
each of these changes, which are what happens to real photos on their way to a checker:

  web          scaled to 384 px and saved as JPEG quality 85 (an ordinary web photo)
  jpeg         re-saved as JPEG quality 40
  small        shrunk to 96 px and enlarged again (a small face in a big photo)
  blur         Gaussian blur, sigma 1.5
  messenger    shrunk to 160 px, then JPEG quality 50 (a messaging-app re-upload)
  stage_light  a strong red colour cast with raised contrast (concert lighting)

Writes reports/robustness.json. With --baseline-head it also scores an earlier head
on exactly the same crops.

Usage:
  python ml/scripts/eval_robustness.py --data data/crops --n 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.classifier import FusionHead  # noqa: E402
from ml.scripts.train_fusion import extract, features, metrics  # noqa: E402


def jpeg(img: np.ndarray, quality: int = 40) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def small(img: np.ndarray, side: int = 96) -> np.ndarray:
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale >= 1:
        return img
    tiny = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_CUBIC)


def blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def messenger(img: np.ndarray) -> np.ndarray:
    return jpeg(small(img, 160), 50)


def stage_light(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = x * np.array([0.45, 0.40, 1.25], np.float32)  # BGR: push everything toward red
    mean = x.mean()
    x = np.clip((x - mean) * 1.25 + mean, 0, 1) ** 1.1
    return (x * 255).astype(np.uint8)


def web(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 384 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return jpeg(img, 85)


CHANGES = {"web": web, "jpeg": jpeg, "small": small, "blur": blur, "messenger": messenger, "stage_light": stage_light}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crops")
    ap.add_argument("--head", default="models/fusion_head.npz")
    ap.add_argument("--baseline-head", default="")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    data_dir = ROOT / args.data
    df = pd.read_csv(data_dir / "manifest.csv")
    held = df[df.split.isin(["test", "unseen"])]
    # The same number of crops from every (source, label) cell, so no dataset dominates.
    per_cell = max(1, args.n // held.groupby(["source", "label"]).ngroups)
    sample = pd.concat(g.sample(min(len(g), per_cell), random_state=0) for _, g in held.groupby(["source", "label"]))
    sample = sample.reset_index(drop=True)

    frames = {"clean": sample}
    for name, change in CHANGES.items():
        rows = sample.copy()
        rows["path"] = [str(Path("robust") / name / p) for p in sample.path]
        for src, dst in zip(sample.path, rows.path):
            out = data_dir / dst
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), change(cv2.imread(str(data_dir / src))))
        frames[name] = rows

    heads = {"fusion_head": FusionHead(ROOT / args.head)}
    if args.baseline_head:
        heads["previous_head"] = FusionHead(ROOT / args.baseline_head)
    models = sorted({m for h in heads.values() for m in h.model_ids})
    y = sample.label.to_numpy()
    report: dict = {"n": int(len(sample)), "changes": list(CHANGES)}
    clean_pred: dict[str, np.ndarray] = {}
    for name, rows in frames.items():
        feats = {m: extract(m, rows, data_dir, ROOT / "data" / "features") for m in models}
        for label, head in heads.items():
            p = head.predict(features(feats, head.model_ids, np.arange(len(rows))))
            if name == "clean":
                clean_pred[label] = p >= 0.5
            entry = metrics(y, p)
            entry["verdict_flipped"] = round(float(((p >= 0.5) != clean_pred[label]).mean()), 4)
            report.setdefault(label, {})[name] = entry
    (ROOT / "reports" / "robustness.json").write_text(json.dumps(report, indent=2))
    for label in heads:
        print(label)
        for name, entry in report[label].items():
            print(f"  {name:12s} AUC {entry['auc']:.3f}  accuracy {entry['accuracy']:.3f}  "
                  f"flipped {entry['verdict_flipped']:.3f}")


if __name__ == "__main__":
    main()
