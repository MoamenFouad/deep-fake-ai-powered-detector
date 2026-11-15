"""Benchmark pretrained backbones and train the DeepShield fusion head.

Pipeline
  1. Extract (P(fake), embedding) for every crop with every candidate backbone.
     Results are cached per backbone in data/features/.
  2. Model selection uses the *train* split only, with leave-one-domain-out
     validation: the training sources are grouped into eight domains that each hold
     real and fake faces, and every domain is scored by a head that never saw it.
     This measures what matters for a detector, which is how it copes with a kind
     of fake or photo it was not trained on. The score is the mean per-domain AUC,
     averaged over the original crops and their degraded copies (harsh JPEG,
     down-scaling, blur, noise and colour casts, plus a "shared online" copy).
     Backbones are added greedily while that score improves, then the
     regularisation strength C is tuned the same way.
  3. The head is retrained on all of train (original crops plus two augmented copies
     of each) and its logit is Platt-calibrated on the leave-one-domain-out
     predictions, so the reported probability reflects unfamiliar data.
  4. One evaluation on the held-out splits: the four core test sets, the test sets
     of the newer sources, and the "unseen" sources that are never trained on.
     Writes reports/metrics.json, reports/test_predictions.npz and
     models/fusion_head.npz.

Usage:
  python ml/scripts/train_fusion.py --data data/crops
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.classifier import Backbone, FusionHead, build_features  # noqa: E402

# Deepfake / AI-image classifiers and general vision encoders from the Hugging Face Hub.
CANDIDATES = [
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "prithivMLmods/deepfake-detector-model-v1",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/deepfake_vs_real_image_detection",
    "Wvolf/ViT_Deepfake_Detection",
    "Ateeqq/ai-vs-human-image-detector",
    "Organika/sdxl-detector",
    "openai/clip-vit-base-patch16",
    "facebook/dinov2-small",
]
CORE_SOURCES = ["ffhq_stylegan", "dff", "celebdf", "dfd"]
# Validation domains. Sources that hold only real or only fake faces are paired with
# their counterpart (same photo style and resolution), so every domain has both classes
# and holding one out never leaves half of a matched pair in training.
DOMAINS = {
    "ffhq_stylegan": "ffhq_stylegan", "dff": "dff", "celebdf": "celebdf", "dfd": "dfd",
    "celeba": "celeba_swaps", "faceswap": "celeba_swaps",
    "celebahq": "hq_flux", "flux_faces": "hq_flux",
    "coco": "everyday_sdxl", "flickr": "everyday_sdxl", "sdxl_faces": "everyday_sdxl",
    "pexels": "portraits_t2i", "sfhq_t2i": "portraits_t2i",
}
C_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.02]


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
    return build_features([feats[m]["emb"][idx] for m in models], [feats[m]["p"][idx] for m in models])


# ------------------------------------------------------------------------- models
def cell_weights(df: pd.DataFrame) -> np.ndarray:
    """Give every (source, label) cell the same total weight."""
    counts = df.groupby(["source", "label"])["path"].transform("count").to_numpy()
    w = 1.0 / counts
    return w / w.mean()


def fit_head(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float):
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit((X - mean) / std, y, sample_weight=w)
    return clf, mean, std


def head_logit(model, X: np.ndarray) -> np.ndarray:
    clf, mean, std = model
    return clf.decision_function((X - mean) / std)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lodo_logits(df: pd.DataFrame, X: np.ndarray, C: float) -> np.ndarray:
    """Leave-one-domain-out: every training crop is scored by a head trained without its domain."""
    y = df["label"].to_numpy()
    domain = df["source"].map(DOMAINS).fillna(df["source"]).to_numpy()
    out = np.zeros(len(df))
    for d in np.unique(domain):
        keep = domain != d
        model = fit_head(X[keep], y[keep], cell_weights(df[keep]), C)
        out[~keep] = head_logit(model, X[~keep])
    return out


def lodo_score(df: pd.DataFrame, logits: np.ndarray) -> dict:
    """Mean per-domain AUC on the original crops and on their degraded copies."""
    table = pd.DataFrame({"domain": df["source"].map(DOMAINS).fillna(df["source"]).to_numpy(),
                          "aug": df["aug"].to_numpy(), "y": df["label"].to_numpy(), "logit": logits})
    out = {}
    for name, part in (("clean", table[table.aug == 0]), ("degraded", table[table.aug > 0])):
        if len(part):
            per = {d: float(roc_auc_score(g.y, g.logit)) for d, g in part.groupby("domain") if g.y.nunique() == 2}
            out[name] = {"mean_domain_auc": round(float(np.mean(list(per.values()))), 4),
                         "per_domain_auc": {k: round(v, 4) for k, v in per.items()}}
    out["score"] = round(float(np.mean([v["mean_domain_auc"] for v in out.values()])), 4)
    return out


def platt(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Fit p = sigmoid(a * logit + b)."""
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(logits.reshape(-1, 1), y, sample_weight=w)
    return float(cal.coef_[0, 0]), float(cal.intercept_[0])


# ------------------------------------------------------------------------ metrics
def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, pred)), 4)}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        out.update({
            "auc": round(float(roc_auc_score(y, p)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "eer": round(float(fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]), 4),
        })
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: gap between stated probability and observed fake rate."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return round(float(total), 4)


def report(df: pd.DataFrame, p: np.ndarray) -> dict:
    """Overall metrics plus one entry per source (accuracy only for single-class sources)."""
    y = df["label"].to_numpy()
    out = {"overall": {**metrics(y, p), "ece": ece(y, p)}}
    for src, sub in df.groupby("source"):
        idx = df.index.get_indexer(sub.index)
        out[src] = metrics(sub["label"].to_numpy(), p[idx])
    aucs = [v["auc"] for k, v in out.items() if k != "overall" and "auc" in v]
    if aucs:
        out["mean_source_auc"] = round(float(np.mean(aucs)), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crops")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--backbones", default="", help="Skip selection and use these backbones")
    ap.add_argument("--max-backbones", type=int, default=3)
    ap.add_argument("--out-head", default="models/fusion_head.npz")
    ap.add_argument("--out-report", default="reports/metrics.json")
    ap.add_argument("--baseline-head", default="", help="An earlier fusion head to compare with on the same splits")
    ap.add_argument("--extract-only", action="store_true", help="Only fill the feature cache")
    args = ap.parse_args()

    data_dir = ROOT / args.data
    df = pd.read_csv(data_dir / "manifest.csv")
    if "aug" not in df:
        df["aug"] = 0
    cands = args.backbones.split(",") if args.backbones else args.candidates.split(",")
    feats = {m: extract(m, df, data_dir, ROOT / "data" / "features") for m in cands}
    if args.extract_only:
        return

    y = df["label"].to_numpy()
    train = np.flatnonzero((df.split == "train").to_numpy())
    train_df = df.iloc[train].reset_index(drop=True)
    y_train = y[train]
    is_orig = train_df["aug"].to_numpy() == 0
    w_orig = cell_weights(train_df[is_orig])
    print(f"train={len(train)} (original {int(is_orig.sum())}), test={(df.split == 'test').sum()}, "
          f"unseen={(df.split == 'unseen').sum()}")

    w_train = cell_weights(train_df)

    def validate(models: list[str], C: float) -> dict:
        return lodo_score(train_df, lodo_logits(train_df, features(feats, models, train), C))

    # 1) Zero-shot ranking of each classifier on the original training crops.
    zero_shot = {}
    for m in cands:
        p = feats[m]["p"][train][is_orig]
        if not np.isnan(p).all():
            zero_shot[short(m)] = round(float(roc_auc_score(y_train[is_orig], p, sample_weight=w_orig)), 4)
    print("Zero-shot AUC on train:", json.dumps(zero_shot, indent=1))

    # 2) Greedy backbone selection by leave-one-domain-out AUC.
    selection_log = []
    if args.backbones:
        chosen = cands
    else:
        chosen, best = [], 0.0
        while len(chosen) < args.max_backbones:
            scores = {m: validate(chosen + [m], 0.002)["score"] for m in cands if m not in chosen}
            m, score = max(scores.items(), key=lambda kv: kv[1])
            selection_log.append({"with": [short(c) for c in chosen],
                                  "scores": {short(k): round(v, 4) for k, v in scores.items()}})
            print(f"  add to {[short(c) for c in chosen]}: {selection_log[-1]['scores']}", flush=True)
            if chosen and score < best + 0.003:  # a backbone must earn its memory and time
                break
            chosen, best = chosen + [m], score
    print("Chosen backbones:", chosen)

    # 3) Tune C the same way, then calibrate on the leave-one-domain-out predictions.
    c_scores = {c: validate(chosen, c)["score"] for c in C_GRID}
    best_c = max(c_scores, key=c_scores.get)
    print("Leave-one-domain-out score per C:", c_scores, "->", best_c)
    X_train = features(feats, chosen, train)
    out_of_domain = lodo_logits(train_df, X_train, best_c)
    validation = lodo_score(train_df, out_of_domain)
    cal_a, cal_b = platt(out_of_domain, y_train, w_train)
    p_ood = sigmoid(cal_a * out_of_domain + cal_b)
    y_o, p_o = y_train[is_orig], p_ood[is_orig]
    validation["clean"].update({**metrics(y_o, p_o), "ece": ece(y_o, p_o),
                                "ece_uncalibrated": ece(y_o, sigmoid(out_of_domain[is_orig]))})
    print(f"Platt calibration: a={cal_a:.3f} b={cal_b:.3f}; validation {json.dumps(validation, indent=1)}")

    # 4) Retrain on all of train and evaluate once.
    head = fit_head(X_train, y_train, cell_weights(train_df), best_c)

    def predict(idx: np.ndarray) -> np.ndarray:
        return sigmoid(cal_a * head_logit(head, features(feats, chosen, idx)) + cal_b)

    baseline = FusionHead(ROOT / args.baseline_head) if args.baseline_head else None

    def evaluate(mask: np.ndarray) -> dict:
        idx = np.flatnonzero(mask)
        sub = df.iloc[idx].reset_index(drop=True)
        out = {"fusion_head": report(sub, predict(idx))}
        classifiers = [m for m in chosen if not np.isnan(feats[m]["p"][idx]).all()]
        if classifiers:
            out["ensemble_mean"] = report(sub, np.mean([feats[m]["p"][idx] for m in classifiers], axis=0))
        for m in cands:
            if not np.isnan(feats[m]["p"][idx]).all():
                out[f"zero_shot:{short(m)}"] = report(sub, feats[m]["p"][idx])
        if baseline is not None and all(m in feats for m in baseline.model_ids):
            out["previous_head"] = report(sub, baseline.predict(features(feats, baseline.model_ids, idx)))
        return out

    test = (df.split == "test").to_numpy()
    core = test & df.source.isin(CORE_SOURCES).to_numpy()
    results = {
        "version": 2,
        "chosen_backbones": chosen,
        "C": best_c,
        "calibration": {"a": round(cal_a, 4), "b": round(cal_b, 4)},
        "split_sizes": {"train": int(is_orig.sum()), "train_augmented": int((~is_orig).sum()),
                        "test": int(test.sum()), "unseen": int((df.split == "unseen").sum())},
        "zero_shot_train_auc": zero_shot,
        "selection": selection_log,
        "validation_score_per_C": {str(k): v for k, v in c_scores.items()},
        "leave_one_domain_out": validation,
        # "test" keeps the meaning it had in version 1: the four core datasets.
        "test": evaluate(core),
        "test_new_sources": evaluate(test & ~core),
        "test_all": evaluate(test),
        "unseen": evaluate((df.split == "unseen").to_numpy()),
    }
    Path(ROOT / args.out_report).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out_report).write_text(json.dumps(results, indent=2))

    held = np.flatnonzero(test | (df.split == "unseen").to_numpy())
    held_df = df.iloc[held]
    np.savez(ROOT / "reports" / "test_predictions.npz", y=y[held], p_fusion=predict(held),
             source=held_df["source"].to_numpy().astype(str), split=held_df["split"].to_numpy().astype(str),
             kind=held_df["kind"].to_numpy().astype(str),
             **{f"p_{short(m)}": feats[m]["p"][held] for m in cands})

    clf, mean, std = head
    np.savez(ROOT / args.out_head, mean=mean.astype(np.float32), std=std.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             cal_a=np.float32(cal_a), cal_b=np.float32(cal_b), model_ids=np.array(chosen))
    for name in ("test", "test_new_sources", "unseen"):
        print(name, json.dumps(results[name]["fusion_head"]["overall"]))
    print(f"Saved head -> {args.out_head}, report -> {args.out_report}")


if __name__ == "__main__":
    main()
