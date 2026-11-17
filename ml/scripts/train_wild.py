"""Version 3 heads: retrain on 2025-2026 generators and real photos from the wild.

Version 2 was trained on face crops from research datasets (StyleGAN, FLUX.1, SDXL,
face swaps). Tested on images people actually share in 2026 (GPT-image, Nano Banana,
Midjourney 7, Flux.2, Seedream, Sora and Veo frames, and real photos posted on Reddit)
it missed about a third of the fakes. This script adds that kind of data and trains:

  face head   the same 1,154 features as before (two frozen backbones on the face crop),
              trained on the version 2 crops plus faces from the new images.
  whole head  the same two backbones on the whole picture, so the background, lighting
              and rendering of the scene count as evidence too.
  stacker     combines the two calibrated scores into one probability.

Inputs come from `extract_wild.py` (one feature record per image) and its metadata CSVs.
Data (all public, all on the Hugging Face Hub):
  ComplexDataLab/OpenFake, CC BY-NC 4.0: train split (fakes from 50+ generators, real
      LAION and Pexels photos), test split and the in-the-wild Reddit split
  detection-datasets/coco: everyday photos of people (COCO shards not used in version 2)

Splits: Reddit images are split 50/50 by file hash. OpenFake test, COCO shards 10-13 and
half of Reddit are only ever tested. 20% of the training images (by hash) are held out
to choose C, calibrate and fit the stacker.

    python -m ml.scripts.train_wild --feats data/wild/feats.pkl --meta "data/wild/meta_*.csv"
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from app.services.classifier import FusionHead, build_features
from ml.scripts.train_fusion import ece, fit_head, head_logit, metrics, platt, sigmoid

ROOT = Path(__file__).resolve().parents[2]
MODEL_IDS = ["buildborderless/CommunityForensics-DeepfakeDet-ViT", "prithivMLmods/deepfake-detector-model-v1"]
C_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def _h(s: str, mod: int = 100) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_wild(feats_path: str, meta_glob: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    meta = pd.concat([pd.read_csv(f) for f in glob.glob(meta_glob)]).drop_duplicates("file").set_index("file")
    recs = pickle.load(open(feats_path, "rb"))
    rows, face, whole, old_p = [], [], [], []
    for path, r in recs.items():
        name = Path(path).name
        if name not in meta.index or "whole_feat" not in r:
            continue
        m = meta.loc[name]
        y = int(m["label"] == "fake")
        if m["split"] == "reddit":
            source, split = "reddit", "test" if name[0] in "01234567" else "train"
        elif m["split"] == "core":
            source, split = "openfake_test", "test"
        elif m["split"] == "coco":
            source, split = "coco_wild", "test"
        elif m["split"] == "cocotrain":
            source, split = "coco_wild", "train"
        else:  # OpenFake train
            source, split = ("openfake" if y else str(m["model"])), "train"
        if split == "train" and _h(name) < 20:
            split = "val"
        has_face = "face_feat" in r
        i = int(np.argmax(r["face_area"])) if has_face else 0
        rows.append(dict(path=name, source=source, label=y, split=split, generator=str(m["model"]),
                         has_face=has_face))
        face.append(r["face_feat"][i] if has_face else np.full(1154, np.nan, np.float32))
        whole.append(r["whole_feat"])
        old_p.append(float(r["face_p"].max()) if has_face else np.nan)
    return pd.DataFrame(rows), np.array(face), np.array(whole), np.array(old_p)


def load_crops(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(data_dir / "crops" / "manifest.csv")
    if "aug" not in df:
        df["aug"] = 0
    parts = {}
    for m in MODEL_IDS:
        z = np.load(data_dir / "features" / f"{m.split('/')[-1]}.npz", allow_pickle=True)
        index = {p: i for i, p in enumerate(z["paths"])}
        idx = np.array([index[p] for p in df["path"]])
        parts[m] = (z["emb"][idx], z["p"][idx])
    X = build_features([parts[m][0] for m in MODEL_IDS], [parts[m][1] for m in MODEL_IDS])
    return df, X


def weights(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby(["source", "label"])["path"].transform("count").to_numpy()
    w = 1.0 / counts
    return w / w.mean()


def auc(y, p) -> float:
    return round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) == 2 else float("nan")


def per_source(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["label"].to_numpy()
    out = {"overall": {**metrics(y, p), "ece": ece(y, p)}}
    for src, sub in df.groupby("source"):
        idx = df.index.get_indexer(sub.index)
        out[src] = metrics(sub["label"].to_numpy(), p[idx])
    fakes = df[df.label == 1]
    out["recall_per_generator"] = {
        g: round(float((p[df.index.get_indexer(s.index)] >= 0.5).mean()), 3)
        for g, s in fakes.groupby("generator") if len(s) >= 10}
    reals = df[df.label == 0]
    out["false_alarm_per_source"] = {
        g: round(float((p[df.index.get_indexer(s.index)] >= 0.5).mean()), 3) for g, s in reals.groupby("source")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--wild-weight", type=float, default=1.0, help="Total weight of new images vs old crops")
    args = ap.parse_args()

    wild, Fw, Ww, old_p = load_wild(args.feats, args.meta)
    crops, Xc = load_crops(ROOT / args.data)
    print(wild.groupby(["split", "source", "label"]).size().to_string())

    old_head = FusionHead(ROOT / "models" / "fusion_head.npz")

    # ---- face head: version 2 crops + faces from the new images
    c_tr = (crops.split == "train").to_numpy() & np.array([_h(p) >= 15 for p in crops.path])
    c_va = (crops.split == "train").to_numpy() & (crops.aug == 0).to_numpy() & ~c_tr
    wf = wild.has_face.to_numpy()
    w_tr, w_va, w_te = (wild.split == "train").to_numpy() & wf, (wild.split == "val").to_numpy() & wf, \
        (wild.split == "test").to_numpy()

    Xf = np.vstack([Xc[c_tr], Fw[w_tr]])
    yf = np.concatenate([crops.label.to_numpy()[c_tr], wild.label.to_numpy()[w_tr]])
    wc, ww = weights(crops[c_tr]), weights(wild[w_tr])
    wf_all = np.concatenate([wc / wc.sum(), args.wild_weight * ww / ww.sum()])
    wf_all = wf_all / wf_all.mean()

    def face_val(model) -> tuple[float, float]:
        return (auc(crops.label[c_va], head_logit(model, Xc[c_va])),
                auc(wild.label[w_va], head_logit(model, Fw[w_va])))

    scores = {}
    for C in C_GRID:
        model = fit_head(Xf, yf, wf_all, C)
        a_old, a_new = face_val(model)
        scores[C] = (round((a_old + a_new) / 2, 4), a_old, a_new)
        print(f"face C={C}: old-crops val AUC {a_old}, new-images val AUC {a_new}", flush=True)
    c_face = max(scores, key=lambda c: scores[c][0])
    face_model = fit_head(Xf, yf, wf_all, c_face)
    # Calibrate on both validation sets, equal weight each.
    lv = np.concatenate([head_logit(face_model, Xc[c_va]), head_logit(face_model, Fw[w_va])])
    yv = np.concatenate([crops.label.to_numpy()[c_va], wild.label.to_numpy()[w_va]])
    wv = np.concatenate([weights(crops[c_va]) / c_va.sum(), weights(wild[w_va]) / w_va.sum()])
    fa, fb = platt(lv, yv, wv / wv.mean())

    def face_p(X: np.ndarray) -> np.ndarray:
        return sigmoid(fa * head_logit(face_model, X) + fb)

    # ---- whole-image head: new images only
    g_tr, g_va = (wild.split == "train").to_numpy(), (wild.split == "val").to_numpy()
    ws = {}
    for C in C_GRID:
        model = fit_head(Ww[g_tr], wild.label.to_numpy()[g_tr], weights(wild[g_tr]), C)
        ws[C] = auc(wild.label[g_va], head_logit(model, Ww[g_va]))
        print(f"whole C={C}: val AUC {ws[C]}", flush=True)
    c_whole = max(ws, key=ws.get)
    whole_model = fit_head(Ww[g_tr], wild.label.to_numpy()[g_tr], weights(wild[g_tr]), c_whole)
    wa, wb = platt(head_logit(whole_model, Ww[g_va]), wild.label.to_numpy()[g_va], weights(wild[g_va]))

    def whole_p(X: np.ndarray) -> np.ndarray:
        return sigmoid(wa * head_logit(whole_model, X) + wb)

    # ---- stacker on the validation images that have a face
    sv = g_va & wf
    Z = np.stack([logit(face_p(Fw[sv])), logit(whole_p(Ww[sv]))], 1)
    st = LogisticRegression(C=1.0).fit(Z, wild.label.to_numpy()[sv], sample_weight=weights(wild[sv]))
    s_face, s_whole, s_b = float(st.coef_[0, 0]), float(st.coef_[0, 1]), float(st.intercept_[0])
    print(f"stacker: {s_face:.3f}*face + {s_whole:.3f}*whole + {s_b:.3f}")

    def combined(Xface: np.ndarray, Xwhole: np.ndarray) -> np.ndarray:
        return sigmoid(s_face * logit(face_p(Xface)) + s_whole * logit(whole_p(Xwhole)) + s_b)

    # ---- evaluate on images never trained on
    te = wild[w_te].reset_index(drop=True)
    tf = te.has_face.to_numpy()
    te_f = te[tf].reset_index(drop=True)
    Fte, Wte = Fw[w_te][tf], Ww[w_te][tf]
    results = {
        "version": 3,
        "data": {k: int(v) for k, v in wild.groupby("split").size().items()},
        "C": {"face": c_face, "whole": c_whole},
        "face_validation": {str(k): v for k, v in scores.items()},
        "whole_validation": {str(k): v for k, v in ws.items()},
        "calibration": {"face": [fa, fb], "whole": [wa, wb]},
        "stacker": {"face": s_face, "whole": s_whole, "bias": s_b},
        "wild_test": {
            "v2_face_head": per_source(te_f, old_p[w_te][tf]),
            "v3_face_head": per_source(te_f, face_p(Fte)),
            "v3_whole_image": per_source(te_f, whole_p(Wte)),
            "v3_combined": per_source(te_f, combined(Fte, Wte)),
        },
    }
    # ---- the version 2 test sets must not get worse
    for name, mask in (("crops_test", (crops.split == "test").to_numpy()),
                       ("crops_unseen", (crops.split == "unseen").to_numpy())):
        sub = crops[mask].reset_index(drop=True)
        results[name] = {"v2_face_head": per_source(sub.assign(generator=sub.kind), old_head.predict(Xc[mask])),
                         "v3_face_head": per_source(sub.assign(generator=sub.kind), face_p(Xc[mask]))}

    out = ROOT / "reports" / "wild_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    for k, v in results["wild_test"].items():
        print(f"{k:16s} {json.dumps(v['overall'])}")
    for name in ("crops_test", "crops_unseen"):
        for k, v in results[name].items():
            print(f"{name} {k:14s} {json.dumps(v['overall'])}")

    clf, mean, std = face_model
    np.savez(ROOT / "models" / "fusion_head_v3.npz", mean=mean.astype(np.float32), std=std.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             cal_a=np.float32(fa), cal_b=np.float32(fb), model_ids=np.array(MODEL_IDS))
    clf, mean, std = whole_model
    np.savez(ROOT / "models" / "whole_head.npz", mean=mean.astype(np.float32), std=std.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             cal_a=np.float32(wa), cal_b=np.float32(wb), model_ids=np.array(MODEL_IDS),
             stack=np.array([s_face, s_whole, s_b], dtype=np.float32))
    np.savez(ROOT / "reports" / "wild_test_predictions.npz", y=te_f.label.to_numpy(), source=te_f.source.to_numpy(),
             generator=te_f.generator.to_numpy(), p_v2=old_p[w_te][tf], p_face=face_p(Fte), p_whole=whole_p(Wte),
             p_combined=combined(Fte, Wte))
    print("saved models/fusion_head_v3.npz, models/whole_head.npz, reports/wild_metrics.json")


if __name__ == "__main__":
    main()
