"""Feature records for the images from collect_wild.py.

For every image, exactly as the app sees it:
  face_feat    the 1,154 fusion features of the largest face crop (both backbones)
  face_p       the current photo head's probability for that face
  whole_feat   the same 1,154 features for the whole picture, scaled to 1,024 px
Records are saved to one pickle every 25 images, so a stopped run resumes where it left off.

    python -m ml.scripts.extract_wild data/wild/list.txt data/wild/feats.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from app.config import settings
from app.services.analyzer import CROP_MAX_SIDE, WHOLE_MAX_SIDE
from app.services.classifier import Backbone, FusionHead, build_features
from app.services.face_detector import FaceDetector, crop_face
from app.services.imaging import bgr_to_pil, decode_image, fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", help="Text file with one image path per line")
    ap.add_argument("out")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(os.cpu_count() or 2)
    paths = [line.strip() for line in open(args.paths) if line.strip()]
    done = pickle.load(open(args.out, "rb")) if os.path.exists(args.out) else {}
    det = FaceDetector(settings.face_model_path)
    backbones = [Backbone(m) for m in settings.model_ids]
    head = FusionHead(settings.head_path)

    def features(images):
        probs, embs = zip(*(b.forward(images) for b in backbones))
        return build_features(list(embs), list(probs))

    t0 = time.time()
    for n, path in enumerate(paths):
        if path in done:
            continue
        try:
            img = decode_image(open(path, "rb").read())
        except Exception as exc:
            print("skip", path, exc)
            continue
        rec = {}
        faces = sorted(det.detect(img), key=lambda f: f.w * f.h, reverse=True)[:1]
        with torch.no_grad():
            if faces:
                crop = bgr_to_pil(fit(crop_face(img, faces[0], settings.face_margin), CROP_MAX_SIDE))
                f = features([crop])
                rec.update(face_feat=f.astype(np.float32), face_p=head.predict(f),
                           face_area=np.array([faces[0].w * faces[0].h]))
            rec["whole_feat"] = features([bgr_to_pil(fit(img, WHOLE_MAX_SIDE))])[0].astype(np.float32)
        done[path] = rec
        if n % 25 == 0:
            pickle.dump(done, open(args.out, "wb"))
            print(n, len(paths), f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(done, open(args.out, "wb"))
    print("done", len(done))


if __name__ == "__main__":
    main()
