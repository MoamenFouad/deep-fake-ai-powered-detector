"""Collect 2025-2026 images with people for the version 3 heads.

Images are read straight from Parquet row groups on the Hugging Face Hub (HTTP range
requests), so nothing larger than one row group is ever downloaded. Only images whose
caption mentions a person and in which YuNet finds a face of at least 64 px are kept.

Sources:
  openfake-train   ComplexDataLab/OpenFake core/train: fakes from 50+ generators,
                   real LAION and Pexels photos (CC BY-NC 4.0)
  openfake-test    ComplexDataLab/OpenFake core/test: newer generators, test only
  reddit           ComplexDataLab/OpenFake reddit/test: real photos and AI images
                   posted on Reddit, the closest thing to what people actually check
  coco             detection-datasets/coco shards 10-13 (test) or 3-9 (train); version 2
                   used shards 0-2 for its face crops

    python -m ml.scripts.collect_wild reddit --row-groups 80 --seed 1
    python -m ml.scripts.collect_wild coco --shards 10:14 --tag coco --cap 700

Each run appends to data/wild/meta_<tag>.csv and writes images to data/wild/img/.
A row group can hold 100 MB of images, so run long collections in several processes
(--start) to keep memory flat.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "wild"
OPENFAKE = "datasets/ComplexDataLab/OpenFake"
PERSON = re.compile(
    r"\b(wom[ae]n|m[ae]n|person|people|girl|boy|selfie|portrait|face|lady|guy|child|kid|teen\w*|actor|actress|"
    r"he|she|her|his|bride|groom|couple|family|student|worker|soldier|player|singer)\b", re.I)
FIELDS = ["file", "split", "label", "model", "w", "h", "face_px", "n_faces", "fmt"]


def detector(threshold: float):
    return cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                     (320, 320), threshold)


def largest_face(det, im: Image.Image) -> tuple[float, int]:
    a = np.asarray(im)[:, :, ::-1]
    w, h = im.size
    sc = min(1.0, 1024 / max(w, h))
    small = cv2.resize(a, (int(w * sc), int(h * sc))) if sc < 1 else np.ascontiguousarray(a)
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(small)
    if faces is None:
        return 0.0, 0
    return max(min(f[2], f[3]) for f in faces) / sc, len(faces)


def save(data: bytes, im: Image.Image, fmt: str | None, prefix: str = "") -> str:
    ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(fmt or "")
    if ext is None:  # anything else is stored losslessly as PNG
        ext, buf = "png", io.BytesIO()
        im.save(buf, "PNG")
        data = buf.getvalue()
    name = f"{prefix}{hashlib.sha1(data).hexdigest()[:16]}.{ext}"
    (OUT / "img" / name).write_bytes(data)
    return name


def openfake(args, fs, writer) -> None:
    split = {"openfake-train": "train", "openfake-test": "core", "reddit": "reddit"}[args.source]
    pattern = {"train": "core/train-", "core": "core/test-", "reddit": "reddit/test-"}[split]
    shards = sorted(p for p in fs.ls(OPENFAKE + "/" + pattern.split("/")[0], detail=False) if pattern in p)
    per_shard = 4 if split == "train" else 47
    jobs = [(s, g) for s in shards for g in range(per_shard)]
    random.Random(args.seed).shuffle(jobs)
    det, kept = detector(0.8), 0
    for k, (shard, g) in enumerate(jobs[: args.row_groups]):
        if k < args.start:
            continue
        f = pq.ParquetFile(fs.open(shard, block_size=8 << 20))
        if g >= f.metadata.num_row_groups:
            continue
        for r in f.read_row_group(g, columns=["image", "label", "model", "prompt"]).to_pylist():
            if not r["prompt"] or not PERSON.search(r["prompt"]):
                continue
            data = r["image"]["bytes"]
            try:
                im = Image.open(io.BytesIO(data))
                fmt = im.format
                im = im.convert("RGB")
            except Exception:
                continue
            px, n = largest_face(det, im)
            if px < 64:
                continue
            writer.writerow([save(data, im, fmt), split, r["label"], r["model"], *im.size, int(px), n, fmt])
            kept += 1
        print(f"{k + 1}/{args.row_groups} {shard.split('/')[-1]} rg{g} kept={kept}", flush=True)


def coco(args, fs, writer) -> None:
    a, b = (int(x) for x in args.shards.split(":"))
    files = sorted(p for p in fs.ls("datasets/detection-datasets/coco/data", detail=False) if "train-" in p)[a:b]
    det, kept = detector(0.85), 0
    for path in files:
        f = pq.ParquetFile(fs.open(path, block_size=8 << 20))
        for g in range(f.metadata.num_row_groups):
            for r in f.read_row_group(g, columns=["image", "objects"]).to_pylist():
                if 0 not in r["objects"]["category"]:  # category 0 = person
                    continue
                data = r["image"]["bytes"]
                im = Image.open(io.BytesIO(data)).convert("RGB")
                px, n = largest_face(det, im)
                if px < 64:
                    continue
                writer.writerow([save(data, im, "JPEG", "coco_"), args.tag, "real", "coco", *im.size, int(px), n,
                                 "JPEG"])
                kept += 1
                if kept >= args.cap:
                    return
            print(f"{path.split('/')[-1]} rg{g} kept={kept}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["openfake-train", "openfake-test", "reddit", "coco"])
    ap.add_argument("--row-groups", type=int, default=40)
    ap.add_argument("--start", type=int, default=0, help="Skip the first N row groups (to resume)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", default="10:14", help="COCO shard range")
    ap.add_argument("--tag", default="", help="Name of the metadata file (meta_<tag>.csv)")
    ap.add_argument("--cap", type=int, default=700)
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.source}_{args.seed}"
    meta = OUT / f"meta_{tag}.csv"
    new = not meta.exists()
    with open(meta, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(FIELDS)
        (coco if args.source == "coco" else openfake)(args, HfFileSystem(), writer)


if __name__ == "__main__":
    main()
