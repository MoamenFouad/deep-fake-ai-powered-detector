"""Extend the DeepShield dataset with newer generators and in-the-wild photos.

`prepare_data.py` builds the four core datasets. This script appends to the same
manifest, in three parts:

1. More training and test data (split 80/20 by file, never by crop):
     fake  flux_faces   FLUX.1-dev faces generated from FFHQ / CelebA-HQ captions
           sdxl_faces   Stable Diffusion XL faces generated from FFHQ captions
           sfhq_t2i     SFHQ-T2I 1024 px portraits (FLUX.1-dev, FLUX.1-pro, SDXL)
           faceswap     face swaps on CelebA photos
     real  celeba       CelebA photos (same size and format as the face swaps)
           celebahq     CelebA-HQ 1024 px photos
           coco         faces found in everyday COCO photos
           flickr       faces found in Flickr30k photos of people
           pexels       faces found in Pexels stock photos (professional lighting and colour grading)
2. "Unseen" sources that are only ever tested, never trained on (split "unseen"):
     fake  midjourney, stylegan3, collabdiff, heygen (talking-head video frames)
     real  lfw, youtube (the real clips that ship with the HeyGen set)
     both  wfir (whichfaceisreal.com: StyleGAN faces vs photos)
3. Two augmented copies of every training crop, so the head does not lean on
   resolution, compression or colour:
     aug/  one or two harsh degradations (JPEG 30-75, down-scaling to 64-160 px and
           back, blur, noise, colour casts)
     web/  what a photo looks like after it has been shared online: scaled down to
           160-512 px and saved as JPEG quality 70-95

Every image goes through the app's own face detector and crop.

Usage:
  python ml/scripts/extend_data.py --out data/crops            # parts 1 and 2
  python ml/scripts/extend_data.py --out data/crops --augment  # part 3
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.face_detector import FaceDetector, crop_face  # noqa: E402

DF40 = "ManhQuangAI/df-40-test-full"
COLUMNS = ["path", "source", "split", "label", "group", "video", "face_found", "kind", "aug"]

# Crops per source. "split" sources are divided 80/20 into train and test.
BUDGET = {
    "flux_faces": 500, "sdxl_faces": 500, "sfhq_t2i": 1000, "faceswap": 500,
    "celeba": 500, "celebahq": 700, "coco": 650, "flickr": 650, "pexels": 700,
    "midjourney": 400, "stylegan3": 300, "collabdiff": 300, "heygen": 300,
    "lfw": 500, "youtube": 300, "wfir": 600,
}
UNSEEN = {"midjourney", "stylegan3", "collabdiff", "heygen", "lfw", "youtube", "wfir"}


def _hash(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16)


def _split(source: str, key: str) -> str:
    if source in UNSEEN:
        return "unseen"
    return "test" if _hash(f"{source}:{key}") % 5 == 0 else "train"


def _decode(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    try:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None


# ------------------------------------------------------------------------ readers
def parquet_images(repo: str, file: str, column: str = "image", seed: int = 0) -> Iterator[tuple[str, bytes, dict]]:
    """Yield (row key, image bytes, row) reading one row group at a time, in random group order."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    pf = pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}/{file}"))
    groups = list(range(pf.metadata.num_row_groups))
    random.Random(seed).shuffle(groups)
    for g in groups:
        for i, row in enumerate(pf.read_row_group(g).to_pylist()):
            cell = row[column]
            data = cell["bytes"] if isinstance(cell, dict) else cell
            if data:
                yield f"{Path(file).stem}_g{g}_{i}", data, row


def zip_images(repo: str, file: str, keep, seed: int = 0) -> Iterator[tuple[str, bytes]]:
    """Yield (member name, bytes) for the zip members accepted by `keep`, in random order.

    The archive is downloaded once to the Hugging Face cache: thousands of small
    range reads against a remote zip are far slower than one large download.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    z = zipfile.ZipFile(hf_hub_download(repo, file, repo_type="dataset"))
    names = sorted(n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")) and keep(n))
    random.Random(seed).shuffle(names)
    for n in names:
        try:
            yield n, z.read(n)
        except (zipfile.BadZipFile, OSError) as exc:  # one bad member should not stop the build
            print(f"    skipped {n}: {exc}")


# ------------------------------------------------------------------------- writer
class Writer:
    def __init__(self, out: Path, detector: FaceDetector):
        self.out = out
        self.detector = detector
        manifest = out / "manifest.csv"
        self.df = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame(columns=COLUMNS)
        for col, default in (("kind", ""), ("aug", 0)):
            if col not in self.df:
                self.df[col] = default
        self.rows: list[dict] = []
        self.paths = set(self.df["path"])

    def count(self, source: str) -> int:
        return int((self.df.source == source).sum()) + sum(r["source"] == source for r in self.rows)

    def add(self, image: np.ndarray | None, source: str, label: int, kind: str, key: str, group: str = "",
            video: str = "", min_face: int = 0, min_score: float = 0.0) -> bool:
        """Detect the largest face, crop it like the app does, and record it."""
        if image is None or image.size == 0:
            return False
        faces = self.detector.detect(image, max_faces=1)
        if faces and (min(faces[0].w, faces[0].h) < min_face or faces[0].score < min_score):
            return False
        if not faces and min_face:
            return False
        crop = crop_face(image, faces[0]) if faces else image
        if crop.size == 0:
            return False
        split = _split(source, group or key)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[-80:]
        rel = Path(source) / split / ("fake" if label else "real") / f"{name}.png"
        if str(rel) in self.paths:  # already built by an earlier run
            return False
        self.paths.add(str(rel))
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        self.rows.append({"path": str(rel), "source": source, "split": split, "label": label,
                          "group": group or key, "video": video, "face_found": int(bool(faces)),
                          "kind": kind, "aug": 0})
        return True

    def save(self) -> None:
        if self.rows:
            self.df = pd.concat([self.df, pd.DataFrame(self.rows)], ignore_index=True)
            self.rows = []
        self.df[COLUMNS].to_csv(self.out / "manifest.csv", index=False)


def fill(w: Writer, source: str, items: Iterator, budget: int, **kw) -> None:
    """Add crops from `items` = (key, image bytes, label, kind[, group]) until the budget is met."""
    have = {0: 0, 1: 0}
    for item in items:
        key, data, label, kind = item[:4]
        group = item[4] if len(item) > 4 else ""
        if sum(have.values()) >= budget:
            break
        have[label] += w.add(_decode(data), source, label, kind, key, group=group, **kw)
    w.save()
    print(f"  {source}: real {have[0]}, fake {have[1]}", flush=True)


# ------------------------------------------------------------------------ sources
def flux_faces(w: Writer, n: int) -> None:
    half = n // 2
    a = parquet_images("bitmind/ffhq-256___FLUX.1-dev_training_faces", "data/train-00000-of-00010.parquet")
    b = parquet_images("bitmind/celeb-a-hq___FLUX.1-dev_training_faces", "data/train-00000-of-00005.parquet")
    fill(w, "flux_faces", ((f"ffhq_{k}", d, 1, "flux_dev") for k, d, _ in a), half)
    fill(w, "flux_faces", ((f"celeb_{k}", d, 1, "flux_dev") for k, d, _ in b), n - half)


def sdxl_faces(w: Writer, n: int) -> None:
    rows = parquet_images("bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces",
                          "base_transforms/train-00000-of-00008.parquet")
    fill(w, "sdxl_faces", ((k, d, 1, "sdxl") for k, d, _ in rows), n)


def sfhq_t2i(w: Writer, n: int) -> None:
    kinds = {"FLUX1_dev": "flux_dev", "FLUX1_pro": "flux_pro", "SDXL": "sdxl"}
    members = zip_images("bitmind/SyntheticFacesHighQuality-T2I", "small-sample.zip", lambda name: True)
    fill(w, "sfhq_t2i", ((m, d, 1, kinds.get(m.split("_image_")[0], "t2i")) for m, d in members), n, min_face=64)


def faceswap(w: Writer, n: int) -> None:
    rows = parquet_images("bitmind/face-swap", "data/face_train.parquet")
    fill(w, "faceswap", ((r["filename"], d, 1, "faceswap") for _, d, r in rows), n)


def celeba(w: Writer, n: int) -> None:
    members = zip_images(DF40, "CollabDiff.zip", lambda name: "/real/" in name)
    fill(w, "celeba", ((m, d, 0, "celeba") for m, d in members), n)


def celebahq(w: Writer, n: int) -> None:
    rows = parquet_images("bitmind/celeb-a-hq", "data/train-00000-of-00006.parquet")
    fill(w, "celebahq", ((k, d, 0, "celebahq") for k, d, _ in rows), n, min_face=64)


def coco(w: Writer, n: int) -> None:
    shards = ("data/train-00000-of-00040-67e35002d152155c.parquet",
              "data/train-00001-of-00040-2c2b33b9504aa843.parquet",
              "data/train-00002-of-00040-58e30306870b1de7.parquet")

    def people() -> Iterator:
        for shard in shards:
            for _, data, row in parquet_images("detection-datasets/coco", shard):
                if 0 in row["objects"]["category"]:  # category 0 = person
                    yield f"coco_{row['image_id']}", data, 0, "coco"
    fill(w, "coco", people(), n, min_face=56, min_score=0.85)


def flickr(w: Writer, n: int) -> None:
    members = zip_images("nlphuji/flickr30k", "flickr30k-images.zip", lambda name: True)
    fill(w, "flickr", ((m, d, 0, "flickr") for m, d in members), n, min_face=56, min_score=0.85)


def pexels(w: Writer, n: int) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    repo, api, rng = "ujin-song/pexels-image-1M", HfApi(), random.Random(0)
    folders = sorted(i.path for i in api.list_repo_tree(repo, path_in_repo="images", repo_type="dataset"))
    rng.shuffle(folders)

    def photos() -> Iterator:
        for folder in folders:
            files = sorted(i.path for i in api.list_repo_tree(repo, path_in_repo=folder, repo_type="dataset"))
            for file in rng.sample(files, min(40, len(files))):
                local = Path(hf_hub_download(repo, file, repo_type="dataset"))
                data = local.read_bytes()
                local.resolve().unlink(missing_ok=True)  # full-size photos are large; keep only the crop
                yield file, data, 0, "pexels"
    fill(w, "pexels", photos(), n, min_face=160, min_score=0.9)


def midjourney(w: Writer, n: int) -> None:
    members = zip_images(DF40, "MidJourney.zip", lambda name: "/fake/" in name)
    fill(w, "midjourney", ((m, d, 1, "midjourney") for m, d in members), n, min_face=64)


def stylegan3(w: Writer, n: int) -> None:
    rows = parquet_images("34data/StyleGAN3_1024x1024_images", "data.parquet")
    fill(w, "stylegan3", ((r["filename"], d, 1, "stylegan3") for _, d, r in rows), n)


def collabdiff(w: Writer, n: int) -> None:
    members = zip_images(DF40, "CollabDiff.zip", lambda name: "/fake/" in name)
    fill(w, "collabdiff", ((m, d, 1, "collabdiff") for m, d in members), n)


def _clip_frames(label_dir: str, per_clip: int = 3) -> Iterator[tuple[str, bytes, str]]:
    """A few frames from each HeyGen-set clip; the clip is the leakage group."""
    seen: dict[str, int] = {}
    for m, d in zip_images(DF40, "heygen_new.zip", lambda name: f"/{label_dir}/" in name):
        clip = m.split("/")[-2]
        if seen.get(clip, 0) < per_clip:
            seen[clip] = seen.get(clip, 0) + 1
            yield m, d, clip


def heygen(w: Writer, n: int) -> None:
    fill(w, "heygen", ((m, d, 1, "heygen", clip) for m, d, clip in _clip_frames("fake")), n, min_face=64)


def youtube(w: Writer, n: int) -> None:
    fill(w, "youtube", ((m, d, 0, "youtube", clip) for m, d, clip in _clip_frames("real")), n, min_face=64)


def lfw(w: Writer, n: int) -> None:
    rows = parquet_images("bitmind/lfw", "data/train-00000-of-00001.parquet")
    # One photo per person keeps a few heavily photographed people from dominating.
    seen: set[str] = set()

    def one_each() -> Iterator:
        for _, d, r in rows:
            person = r["filename"].rsplit("_", 1)[0]
            if person not in seen:
                seen.add(person)
                yield r["filename"], d, 0, "lfw", person
    fill(w, "lfw", one_each(), n)


def wfir(w: Writer, n: int) -> None:
    half = n // 2
    real = zip_images(DF40, "whichfaceisreal.zip", lambda name: "/real/" in name)
    fake = zip_images(DF40, "whichfaceisreal.zip", lambda name: "/fake/" in name)
    fill(w, "wfir", ((f"real_{m}", d, 0, "wfir_real") for m, d in real), half)
    fill(w, "wfir", ((f"fake_{m}", d, 1, "wfir_stylegan") for m, d in fake), n - half)


SOURCES = {f.__name__: f for f in (
    flux_faces, sdxl_faces, sfhq_t2i, faceswap, celeba, celebahq, coco, flickr, pexels,
    midjourney, stylegan3, collabdiff, heygen, youtube, lfw, wfir)}

CORE_KINDS = {"ffhq_stylegan": ("ffhq", "stylegan"), "celebdf": ("celebdf_real", "celebdf_swap"),
              "dfd": ("dfd_real", "dfd_swap")}


def core_kind(row) -> str:
    """Kind label for the four core datasets (they were built before the kind column existed)."""
    fake = int(row.label) == 1
    if row.source in CORE_KINDS:
        return CORE_KINDS[row.source][fake]
    if not fake:
        return "imdb_wiki"
    name = Path(row.path).name
    for prefix, kind in (("text2img", "sd_text2img"), ("inpainting", "sd_inpainting"), ("insight", "insightface")):
        if name.startswith(prefix):
            return kind
    return "sd_text2img"


# ------------------------------------------------------------------- augmentation
def _jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 75)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _downscale(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    side = rng.randint(64, 160)
    if max(h, w) <= side:
        return img
    scale = side / max(h, w)
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.0))


def _noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 10), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _colour(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Colour cast (stage lights, filters), gamma, contrast and saturation."""
    x = img.astype(np.float32) / 255.0
    gains = np.array([rng.uniform(0.45, 1.5) for _ in range(3)], np.float32)
    x = x * (gains / gains.max() * rng.uniform(0.85, 1.15))
    x = np.clip(x, 0, 1) ** rng.uniform(0.6, 1.6)
    mean = x.mean()
    x = (x - mean) * rng.uniform(0.7, 1.4) + mean
    grey = x.mean(axis=2, keepdims=True)
    x = grey + (x - grey) * rng.uniform(0.3, 1.5)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


AUGMENTATIONS = {"jpeg": _jpeg, "downscale": _downscale, "blur": _blur, "noise": _noise, "colour": _colour}


def augment_image(img: np.ndarray, key: str) -> np.ndarray:
    """One or two random degradations, the same every time for the same key."""
    rng = random.Random(_hash(key))
    for name in rng.sample(sorted(AUGMENTATIONS), rng.choice([1, 2])):
        img = AUGMENTATIONS[name](img, rng)
    return img


def web_image(img: np.ndarray, key: str) -> np.ndarray:
    """A crop as it looks after being shared online: smaller, and re-saved as JPEG."""
    rng = random.Random(_hash("web:" + key))
    h, w = img.shape[:2]
    scale = rng.randint(160, 512) / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


# (manifest "aug" value, folder, function)
COPIES = ((1, "aug", augment_image), (2, "web", web_image))


def augment(w: Writer) -> None:
    df = w.df
    done = set(df.loc[df.aug > 0, "path"])
    base = df[(df.split == "train") & (df.aug == 0)]
    rows = []
    for row in base.itertuples():
        img = None
        for flag, folder, change in COPIES:
            rel = str(Path(folder) / row.path)
            if rel in done:
                continue
            if img is None:
                img = cv2.imread(str(w.out / row.path))
            if img is None:
                break
            path = w.out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), change(img, row.path))
            rows.append({**{c: getattr(row, c) for c in COLUMNS}, "path": rel, "aug": flag})
    w.rows = rows
    w.save()
    print(f"  augmented copies written: {len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply every budget")
    ap.add_argument("--augment", action="store_true", help="Write the augmented copies of every training crop")
    args = ap.parse_args()

    out = ROOT / args.out
    w = Writer(out, FaceDetector(ROOT / "models" / "face_detection_yunet_2023mar.onnx"))
    missing = w.df.kind.isna() | (w.df.kind == "")
    if missing.any():
        w.df.loc[missing, "kind"] = w.df[missing].apply(core_kind, axis=1)
    w.df["aug"] = w.df["aug"].fillna(0).astype(int)
    w.save()

    if args.augment:
        augment(w)
    else:
        for name in args.sources.split(","):
            budget = max(2, int(BUDGET[name] * args.scale))
            if w.count(name) >= budget:
                print(f"[{name}] already has {w.count(name)} crops, skipping")
                continue
            print(f"[{name}] target {budget} crops", flush=True)
            SOURCES[name](w, budget - w.count(name) if name not in ("flux_faces", "wfir") else budget)
    print(w.df.groupby(["split", "label"]).size().to_string())
    print(f"Done: {len(w.df)} crops -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
