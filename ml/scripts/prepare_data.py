"""Build the DeepShield face-crop dataset from four public sources.

Every image or video frame goes through the *same* face detector and cropping
code the app uses at inference time (app/services/face_detector.py), so the
training distribution matches what the model sees in production.

Sources (all pulled from the Hugging Face Hub with partial/range reads):
  ffhq_stylegan  140k Real and Fake Faces  (FFHQ photos vs StyleGAN faces)
  dff            DeepFakeFace              (IMDB-WIKI photos vs SD text2img / inpainting / InsightFace swaps)
  celebdf        Celeb-DF v2               (YouTube celebrity videos vs face-swap deepfakes)
  dfd            Google DeepFakeDetection  (FaceForensics++ DFD: actor videos vs deepfakes)

Splits never share an identity / source video, to prevent leakage.
Celeb-DF uses its official test list.

Usage:
  python ml/scripts/prepare_data.py --out data/crops --scale 1.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.face_detector import FaceDetector, crop_face  # noqa: E402

VIDEO_REPO = "gonnerthetooner/deepfake_video_data"
FACES_REPO = "TheKernel01/140k-Real-and-Fake-Faces"
DFF_URL = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/{}.zip"

# Number of samples per (split, label). Multiply with --scale.
BUDGET = {
    "ffhq_stylegan": {"train": 600, "test": 250},          # images
    "dff": {"train": 400, "test": 150},                    # images
    "celebdf": {"train": 120, "test": 40},                 # videos (x frames_per_video)
    "dfd": {"train": 60, "test": 25},                      # videos (x frames_per_video)
}
FRAMES_PER_VIDEO = 4


class Writer:
    def __init__(self, out: Path, detector: FaceDetector):
        self.out = out
        self.detector = detector
        self.rows: list[dict] = []

    def add(self, image_bgr: np.ndarray, source: str, split: str, label: int, group: str, name: str,
            video: str = "", require_face: bool = False) -> bool:
        faces = self.detector.detect(image_bgr, max_faces=1)
        if require_face and not faces:
            return False
        crop = crop_face(image_bgr, faces[0]) if faces else image_bgr
        if crop.size == 0:
            return False
        rel = Path(source) / split / ("fake" if label else "real") / f"{name}.png"
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        self.rows.append({"path": str(rel), "source": source, "split": split, "label": label,
                          "group": group, "video": video, "face_found": int(bool(faces))})
        return True

    def save(self) -> None:
        with open(self.out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def _bucket(key: str, test_frac: float = 0.25) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000
    return "test" if h < test_frac * 1000 else "train"


# --------------------------------------------------------------------------- images
def ffhq_stylegan(w: Writer, budget: dict, rng: random.Random) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    plan = {"train": "data/train-00000-of-00006.parquet", "test": "data/test-00000-of-00002.parquet"}
    for split, file in plan.items():
        pf = pq.ParquetFile(fs.open(f"datasets/{FACES_REPO}/{file}"))
        table = pf.read_row_group(0).to_pylist()
        rng.shuffle(table)
        counts = {0: 0, 1: 0}
        for i, row in enumerate(table):
            lab = int(row["label"])
            if counts[lab] >= budget[split]:
                continue
            img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img is not None and w.add(img, "ffhq_stylegan", split, lab, f"ffhq_{split}_{i}", f"{split}_{i}"):
                counts[lab] += 1
            if min(counts.values()) >= budget[split]:
                break
        print(f"  ffhq_stylegan/{split}: {counts}")


def dff(w: Writer, budget: dict, rng: random.Random) -> None:
    from remotezip import RemoteZip

    zips = {name: RemoteZip(DFF_URL.format(name)) for name in ("wiki", "text2img", "inpainting", "insight")}
    # Identities are shared across the four archives; split by identity.
    ids = sorted({n.split("/", 1)[1] for n in zips["wiki"].namelist() if n.endswith(".jpg")})
    rng.shuffle(ids)
    for split in ("train", "test"):
        pool = [i for i in ids if _bucket(i) == split]
        real_ids = pool[: budget[split]]
        fake_ids = pool[budget[split]: 2 * budget[split]]
        fake_kinds = ["text2img", "inpainting", "insight"]
        jobs = [("wiki", rid, 0) for rid in real_ids] + [
            (fake_kinds[k % 3], fid, 1) for k, fid in enumerate(fake_ids)]
        n = 0
        for kind, ident, lab in jobs:
            try:
                data = zips[kind].read(f"{kind}/{ident}")
            except KeyError:
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            name = f"{kind}_{ident.replace('/', '_').removesuffix('.jpg')}"
            if img is not None and w.add(img, "dff", split, lab, ident, name):
                n += 1
        print(f"  dff/{split}: {n} images")


# --------------------------------------------------------------------------- videos
def _frames(path: str, n: int, max_seconds: float = 12.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last = max(1, min(total, int(fps * max_seconds)))
    out = []
    for idx in np.linspace(0, last - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def _download(file: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(VIDEO_REPO, file, repo_type="dataset")


def _process_videos(w: Writer, source: str, items: list[tuple[str, str, int, str]]) -> None:
    """items: (repo_file, split, label, group)"""
    with ThreadPoolExecutor(6) as pool:
        paths = pool.map(lambda it: _download(it[0]), items)
        for (file, split, lab, group), local in zip(items, paths):
            stem = Path(file).stem
            kept = 0
            # Over-sample frames and keep the first few that contain a face.
            for k, frame in enumerate(_frames(local, FRAMES_PER_VIDEO * 3)):
                if kept >= FRAMES_PER_VIDEO:
                    break
                kept += w.add(frame, source, split, lab, group, f"{stem}_f{k}", video=file, require_face=True)
            Path(local).unlink(missing_ok=True)


def celebdf(w: Writer, budget: dict, rng: random.Random) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(VIDEO_REPO, repo_type="dataset")
             if f.startswith("celebdf_v2/") and f.endswith(".mp4")]
    test_list = Path(hf_hub_download(VIDEO_REPO, "celebdf_v2/List_of_testing_videos.txt",
                                     repo_type="dataset")).read_text().split("\n")
    test_set = {"celebdf_v2/" + line.split()[1] for line in test_list if line.strip()}
    # Test identities are excluded from training entirely.
    test_ids = {re.findall(r"id\d+", f)[0] for f in test_set if "id" in Path(f).name}

    def ident(f: str) -> str:
        found = re.findall(r"id\d+", Path(f).name)
        return found[0] if found else Path(f).stem

    items = []
    for split in ("train", "test"):
        for lab in (0, 1):
            if split == "test":
                pool = [f for f in test_set if ("synthesis" in f) == bool(lab)]
            else:
                pool = [f for f in files if f not in test_set and ("synthesis" in f) == bool(lab)
                        and ident(f) not in test_ids]
            rng.shuffle(pool)
            items += [(f, split, lab, ident(f)) for f in pool[: budget[split]]]
    _process_videos(w, "celebdf", items)
    print(f"  celebdf: {len(items)} videos")


def dfd(w: Writer, budget: dict, rng: random.Random) -> None:
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(VIDEO_REPO, repo_type="dataset")
             if f.startswith("google_dfd/") and f.endswith(".mp4")]

    def actor(f: str) -> int:
        return int(Path(f).name[:2])

    items = []
    for lab in (0, 1):
        pool = [f for f in files if ("manipulated" in f) == bool(lab)]
        rng.shuffle(pool)
        for split, actors in (("train", range(1, 21)), ("test", range(21, 29))):
            chosen = [f for f in pool if actor(f) in actors][: budget[split]]
            items += [(f, split, lab, f"actor{actor(f):02d}") for f in chosen]
    _process_videos(w, "dfd", items)
    print(f"  dfd: {len(items)} videos")


SOURCES = {"ffhq_stylegan": ffhq_stylegan, "dff": dff, "celebdf": celebdf, "dfd": dfd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply every budget")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(out, FaceDetector(ROOT / "models" / "face_detection_yunet_2023mar.onnx"))
    print(f"Face detector: {w.detector.backend}")
    for name in args.sources.split(","):
        rng = random.Random(args.seed)
        budget = {k: max(2, int(v * args.scale)) for k, v in BUDGET[name].items()}
        print(f"[{name}] budget per class {budget}")
        SOURCES[name](w, budget, rng)
        w.save()
    print(f"Done: {len(w.rows)} crops -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
