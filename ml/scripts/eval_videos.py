"""End-to-end video evaluation through the production pipeline.

Runs app.services.analyzer.Analyzer (face detection, frame sampling, fusion head,
aggregation) on whole test videos that were never used for training:
  * Celeb-DF v2 official test list
  * Google DFD videos of held-out actors (21-28)

    python ml/scripts/eval_videos.py --per-class 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.services.analyzer import Analyzer  # noqa: E402
from app.services.classifier import DeepfakeClassifier  # noqa: E402
from app.services.face_detector import FaceDetector  # noqa: E402
from ml.scripts.train_fusion import metrics  # noqa: E402

REPO = "gonnerthetooner/deepfake_video_data"


def pick_videos(per_class: int, seed: int) -> list[tuple[str, int, str]]:
    from huggingface_hub import HfApi, hf_hub_download

    rng = random.Random(seed)
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    lines = Path(hf_hub_download(REPO, "celebdf_v2/List_of_testing_videos.txt", repo_type="dataset")).read_text()
    test = ["celebdf_v2/" + ln.split()[1] for ln in lines.splitlines() if ln.strip()]
    out = []
    for lab in (0, 1):
        pool = [f for f in test if ("synthesis" in f) == bool(lab)]
        rng.shuffle(pool)
        out += [(f, lab, "celebdf") for f in pool[:per_class]]
    dfd = [f for f in files if f.startswith("google_dfd/") and f.endswith(".mp4") and int(Path(f).name[:2]) >= 21]
    for lab in (0, 1):
        pool = [f for f in dfd if ("manipulated" in f) == bool(lab)]
        rng.shuffle(pool)
        out += [(f, lab, "dfd") for f in pool[: per_class // 2]]
    return out


def main() -> None:
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s = Settings()
    analyzer = Analyzer(s, FaceDetector(s.face_model_path), DeepfakeClassifier(s.model_ids, s.head_path))
    print("Engine:", analyzer.classifier.engine)
    rows = []
    for i, (file, lab, src) in enumerate(pick_videos(args.per_class, args.seed)):
        local = Path(hf_hub_download(REPO, file, repo_type="dataset"))
        r = analyzer.analyze_video(local.read_bytes(), local.name)
        local.unlink(missing_ok=True)
        rows.append({"video": file, "source": src, "label": lab, "p_fake": r.fake_probability,
                     "verdict": r.verdict, "frames": r.frames_analyzed, "ms": r.processing_ms})
        print(f"[{i + 1}] {src} label={lab} p={r.fake_probability:.3f} {r.verdict} ({r.processing_ms} ms)", flush=True)

    out = ROOT / "reports"
    with open(out / "video_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    y = np.array([r["label"] for r in rows])
    p = np.array([r["p_fake"] for r in rows])
    res = {"overall": metrics(y, p), "engine": analyzer.classifier.engine,
           "mean_seconds_per_video": round(float(np.mean([r["ms"] for r in rows])) / 1000, 2),
           "uncertain_rate": round(float(np.mean([r["verdict"] == "uncertain" for r in rows])), 3)}
    for src in ("celebdf", "dfd"):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        res[src] = metrics(y[idx], p[idx])
    (out / "video_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
