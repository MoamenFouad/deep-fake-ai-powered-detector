"""Benchmark pretrained backbones and train the DeepShield fusion head.

Pipeline
  1. Extract (P(fake), pooled embedding) for every crop with every candidate
     backbone. Results are cached per backbone in data/features/.
  2. Carve a group-wise validation split out of *train* (no identity overlap).
  3. Model selection on validation only: rank backbones and greedily pick the
     combination (size <= --max-backbones) with the best fusion AUC.
  4. Tune the regularisation strength C on validation, retrain on all of train.
  5. Evaluate once on the held-out test split and write reports/metrics.json
     plus models/fusion_head.npz.

Usage:
  python ml/scripts/train_fusion.py --data data/crops
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.classifier import Backbone, _logit  # noqa: E402

CANDIDATES = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/deepfake_vs_real_image_detection",
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "prithivMLmods/deepfake-detector-model-v1",
    "Wvolf/ViT_Deepfake_Detection",
]


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


# ----------------------------------------------------------------------- features
def extract(model_id: str, df: pd.DataFrame, data_dir: Path, cache_dir: Path) -> dict[str, np.ndarray]:
    """Incremental per-crop cache: only crops not seen before are run through the model."""
    cache = cache_dir / f"{short(model_id)}.npz"
    known: dict[str, tuple[float, np.ndarray]] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        known = {str(path): (float(p), e) for path, p, e in zip(z["paths"], z["p"], z["emb"])}
    todo = [p for p in df["path"] if p not in known]

    def save() -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = list(known)
        np.savez_compressed(cache, paths=np.array(paths), p=np.array([known[k][0] for k in paths]),
                            emb=np.stack([known[k][1] for k in paths]))

    if todo:
        print(f"Extracting features with {model_id} for {len(todo)} new crops ...", flush=True)
        bb = Backbone(model_id)
        chunk = 64
        for i in range(0, len(todo), chunk):
            batch = todo[i:i + chunk]
            p, e = bb.forward([Image.open(data_dir / path).convert("RGB") for path in batch])
            known.update({path: (float(pp), ee.astype(np.float32)) for path, pp, ee in zip(batch, p, e)})
            if (i // chunk) % 10 == 0:
                print(f"  {i + len(batch)}/{len(todo)}", flush=True)
                save()  # checkpoint so an interrupted run resumes where it stopped
        del bb
        save()
    return {"p": np.array([known[k][0] for k in df["path"]]), "emb": np.stack([known[k][1] for k in df["path"]])}


def features(feats: dict, models: list[str], idx: np.ndarray) -> np.ndarray:
    embs = [feats[m]["emb"][idx] for m in models]
    logits = np.stack([_logit(feats[m]["p"][idx]) for m in models], axis=1)
    return np.concatenate([*embs, logits], axis=1)


# ------------------------------------------------------------------------- models
def source_weights(df: pd.DataFrame) -> np.ndarray:
    """Give every (source, label) cell the same total weight."""
    counts = df.groupby(["source", "label"])["path"].transform("count").to_numpy()
    w = 1.0 / counts
    return w / w.mean()


def fit_head(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float):
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=4000)
    clf.fit((X - mean) / std, y, sample_weight=w)
    return clf, mean, std


def predict_head(model, X):
    clf, mean, std = model
    return clf.predict_proba((X - mean) / std)[:, 1]


# ------------------------------------------------------------------------ metrics
def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    if len(np.unique(y)) < 2:
        auc, eer = float("nan"), float("nan")
    else:
        fpr, tpr, _ = roc_curve(y, p)
        eer = float(fpr[np.nanargmin(np.abs((1 - tpr) - fpr))])
        auc = float(roc_auc_score(y, p))
    return {
        "auc": round(auc, 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "eer": round(eer, 4),
        "n": int(len(y)),
    }


def report(df: pd.DataFrame, p: np.ndarray) -> dict:
    out = {"overall": metrics(df["label"].to_numpy(), p)}
    for src, sub in df.groupby("source"):
        idx = df.index.get_indexer(sub.index)
        out[src] = metrics(sub["label"].to_numpy(), p[idx])
    # Balanced view: average over sources so no single dataset dominates.
    aucs = [v["auc"] for k, v in out.items() if k != "overall" and np.isfinite(v["auc"])]
    out["mean_source_auc"] = round(float(np.mean(aucs)), 4) if aucs else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crops")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--max-backbones", type=int, default=2)
    ap.add_argument("--out-head", default="models/fusion_head.npz")
    ap.add_argument("--out-report", default="reports/metrics.json")
    ap.add_argument("--extract-only", action="store_true", help="Only fill the feature cache")
    args = ap.parse_args()

    data_dir = ROOT / args.data
    df = pd.read_csv(data_dir / "manifest.csv")
    cands = args.candidates.split(",")
    feats = {m: extract(m, df, data_dir, ROOT / "data" / "features") for m in cands}
    if args.extract_only:
        return

    train_all = df[df.split == "train"].index.to_numpy()
    test = df[df.split == "test"].index.to_numpy()
    groups = (df.loc[train_all, "source"] + ":" + df.loc[train_all, "group"].astype(str)).to_numpy()
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(train_all, groups=groups))
    tr, va = train_all[tr_i], train_all[va_i]
    y = df["label"].to_numpy()
    print(f"train={len(tr)} val={len(va)} test={len(test)}")

    # 1) Zero-shot validation ranking of each backbone.
    val_df = df.loc[va].reset_index(drop=True)
    zero_shot_val = {m: report(val_df, feats[m]["p"][va])["mean_source_auc"] for m in cands}
    print("Zero-shot val mean-source AUC:", json.dumps({short(k): v for k, v in zero_shot_val.items()}, indent=1))

    # 2) Pick the backbone combination with the best fused validation AUC.
    w_tr = source_weights(df.loc[tr])
    best = (None, -1.0)
    for k in range(1, args.max_backbones + 1):
        for combo in itertools.combinations(cands, k):
            head = fit_head(features(feats, list(combo), tr), y[tr], w_tr, C=0.05)
            score = report(val_df, predict_head(head, features(feats, list(combo), va)))["mean_source_auc"]
            print(f"  combo {[short(c) for c in combo]}: val AUC {score}")
            if score > best[1]:
                best = (list(combo), score)
    chosen = best[0]
    print("Chosen backbones:", chosen)

    # 3) Tune C on validation.
    best_c = max([0.005, 0.02, 0.05, 0.2, 1.0], key=lambda c: report(
        val_df, predict_head(fit_head(features(feats, chosen, tr), y[tr], w_tr, c),
                             features(feats, chosen, va)))["mean_source_auc"])
    print("Best C:", best_c)

    # 4) Retrain on all of train, evaluate once on test.
    head = fit_head(features(feats, chosen, train_all), y[train_all], source_weights(df.loc[train_all]), best_c)
    test_df = df.loc[test].reset_index(drop=True)
    p_fusion = predict_head(head, features(feats, chosen, test))

    results = {
        "chosen_backbones": chosen,
        "C": best_c,
        "split_sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(test))},
        "zero_shot_val_auc": {short(k): v for k, v in zero_shot_val.items()},
        "test": {
            **{f"zero_shot:{short(m)}": report(test_df, feats[m]["p"][test]) for m in cands},
            "ensemble_mean": report(test_df, np.mean([feats[m]["p"][test] for m in chosen], axis=0)),
            "fusion_head": report(test_df, p_fusion),
        },
    }
    Path(ROOT / args.out_report).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out_report).write_text(json.dumps(results, indent=2))
    np.savez(ROOT / "reports" / "test_predictions.npz", y=y[test], p_fusion=p_fusion,
             source=test_df["source"].to_numpy(),
             **{f"p_{short(m)}": feats[m]["p"][test] for m in cands})

    clf, mean, std = head
    np.savez(ROOT / args.out_head, mean=mean.astype(np.float32), std=std.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             model_ids=np.array(chosen))
    print(json.dumps(results["test"]["fusion_head"], indent=1))
    print(f"Saved head -> {args.out_head}, report -> {args.out_report}")


if __name__ == "__main__":
    main()
