"""Render evaluation figures (PNG) for the README, docs and slides.

    python ml/scripts/make_figures.py

Reads reports/metrics.json, reports/test_predictions.npz and, when present,
reports/robustness.json. Writes:
  accuracy_by_source.png  share of held-out faces judged correctly, per source
  per_dataset_auc.png     AUC on the four core test sets
  roc.png                 ROC curves on the test and unseen splits
  confusion_matrix.png    verdict counts on the test split
  robustness.png          AUC after each kind of degradation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

# Two series at most per chart: DeepShield in ink, the comparison in grey. Amber marks a weak spot.
INK, GREY, AMBER, MARKER = "#0d0d0d", "#b9bbbf", "#c27400", "#ffd400"
TEXT, TEXT_2, GRID, SURFACE = "#0d0d0d", "#5c5d60", "#e7e7e3", "#ffffff"

CORE = {
    "ffhq_stylegan": "140k Faces\n(StyleGAN)",
    "dff": "DeepFakeFace\n(diffusion, swaps)",
    "celebdf": "Celeb-DF v2\n(video face swap)",
    "dfd": "Google DFD\n(video face swap)",
}
# (metrics split, source key, label) in the order they are drawn, top to bottom.
SOURCES = [
    ("test_all", "ffhq_stylegan", "StyleGAN vs FFHQ photos"),
    ("test_all", "dff", "Stable Diffusion, InsightFace"),
    ("test_all", "flux_faces", "FLUX.1 faces"),
    ("test_all", "sdxl_faces", "Stable Diffusion XL faces"),
    ("test_all", "sfhq_t2i", "FLUX.1 and SDXL portraits"),
    ("test_all", "celebdf", "Celeb-DF v2 face swaps"),
    ("test_all", "dfd", "Google DFD face swaps"),
    ("test_all", "faceswap", "CelebA face swaps"),
    ("test_all", "pexels", "Stock photos (real)"),
    ("test_all", "coco", "Everyday photos, COCO (real)"),
    ("test_all", "flickr", "Everyday photos, Flickr (real)"),
    ("test_all", "celebahq", "Celebrity photos, HQ (real)"),
    ("test_all", "celeba", "Celebrity photos, low-res (real)"),
    ("unseen", "midjourney", "Midjourney faces  *"),
    ("unseen", "stylegan3", "StyleGAN3 faces  *"),
    ("unseen", "collabdiff", "Collaborative Diffusion  *"),
    ("unseen", "wfir", "whichfaceisreal.com  *"),
    ("unseen", "heygen", "HeyGen avatars  *"),
    ("unseen", "youtube", "Real video clips  *"),
    ("unseen", "lfw", "Press photos, LFW (real)  *"),
]
CHANGES = {"clean": "Unchanged", "web": "Web photo\n(384 px, JPEG 85)", "jpeg": "JPEG 40", "small": "96 px face",
           "blur": "Blur", "messenger": "Messenger\nre-upload", "stage_light": "Red stage\nlight"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=180)
    plt.close(fig)


def accuracy_by_source(metrics: dict) -> None:
    rows = [(s, k, label) for s, k, label in SOURCES if k in metrics[s]["fusion_head"]]
    ours = [metrics[s]["fusion_head"][k]["accuracy"] for s, k, _ in rows]
    has_prev = "previous_head" in metrics["test_all"]
    prev = [metrics[s]["previous_head"][k]["accuracy"] for s, k, _ in rows] if has_prev else None
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(rows) + 1.6))
    height = 0.38 if has_prev else 0.6
    ax.barh(y + (height / 2 if has_prev else 0), ours, height=height,
            color=[AMBER if v < 0.7 else INK for v in ours], label="DeepShield v2")
    if has_prev:
        ax.barh(y - height / 2, prev, height=height, color=GREY, label="Previous version")
    for yi, v in zip(y, ours):
        ax.text(v + 0.012, yi + (height / 2 if has_prev else 0), f"{v:.0%}", va="center", fontsize=9.5,
                color=TEXT, fontweight="bold")
    ax.set_yticks(y, [label for _, _, label in rows])
    ax.set_xlim(0, 1.09)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_title("Held-out faces judged correctly, by source")
    ax.set_xlabel("Share on the right side of 50%.   * = source never used in training")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.035), ncols=2, fontsize=10, handlelength=1.2)
    _save(fig, "accuracy_by_source.png")


def per_dataset(metrics: dict) -> None:
    test = metrics["test"]
    keys = [k for k in CORE if k in test["fusion_head"]]
    singles = [v for name, v in test.items() if name.startswith("zero_shot:")]
    series = [("DeepShield v2", [test["fusion_head"][k]["auc"] for k in keys], INK)]
    if "previous_head" in test:
        series.append(("Previous version", [test["previous_head"][k]["auc"] for k in keys], "#7d7f84"))
    series.append(("Best single public model", [max(v[k]["auc"] for v in singles) for k in keys], GREY))
    x = np.arange(len(keys))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, (label, values, colour) in enumerate(series):
        pos = x + (i - (len(series) - 1) / 2) * width
        ax.bar(pos, values, width=width * 0.92, color=colour, label=label)
        for xi, v in zip(pos, values):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=TEXT,
                    fontweight="bold" if i == 0 else "normal")
    ax.axhline(0.5, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(keys) - 0.52, 0.51, "coin flip", fontsize=9, color=TEXT_2, ha="right")
    ax.set_xticks(x, [CORE[k] for k in keys])
    ax.set_ylim(0.4, 1.06)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUC on the test split")
    ax.set_title("The four core test sets")
    ax.legend(loc="lower left", fontsize=9.5, ncols=len(series), bbox_to_anchor=(0, 1.08), handlelength=1.2)
    _save(fig, "per_dataset_auc.png")


def roc(metrics: dict, preds) -> None:
    y, split = preds["y"], preds["split"].astype(str)
    chosen = [m.split("/")[-1] for m in metrics["chosen_backbones"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True)
    for ax, name, title in ((axes[0], "test", "Test split"), (axes[1], "unseen", "Sources never trained on")):
        m = split == name
        curves = [("DeepShield v2", preds["p_fusion"][m], INK, 2.4),
                  ("Average of the two backbones", np.mean([preds[f"p_{c}"][m] for c in chosen], axis=0),
                   "#7d7f84", 1.6),
                  (f"{chosen[0][:18]} alone", preds[f"p_{chosen[0]}"][m], GREY, 1.6)]
        for label, p, colour, lw in curves:
            fpr, tpr, _ = roc_curve(y[m], p)
            ax.plot(fpr, tpr, color=colour, linewidth=lw, label=f"{label}  (AUC {roc_auc_score(y[m], p):.3f})")
        ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(f"{title}, {int(m.sum()):,} faces", fontsize=12)
        ax.set_xlabel("False positive rate (real faces flagged)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("True positive rate (fakes caught)")
    _save(fig, "roc.png")


def confusion(preds) -> None:
    m = preds["split"].astype(str) == "test"
    cm = confusion_matrix(preds["y"][m], (preds["p_fusion"][m] >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("ink", ["#f4f4f0", "#8a8b8e", INK])
    ax.imshow(cm, cmap=ramp, vmin=0)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n{v / cm[i].sum():.0%}", ha="center", va="center", fontsize=13, fontweight="bold",
                color="#ffffff" if v > cm.max() * 0.45 else TEXT)
    ax.set_xticks([0, 1], ["Predicted real", "Predicted fake"])
    ax.set_yticks([0, 1], ["Actually real", "Actually fake"])
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Test split at the 50% threshold")
    _save(fig, "confusion_matrix.png")


def robustness() -> None:
    path = REPORTS / "robustness.json"
    if not path.exists():
        return
    rob = json.loads(path.read_text())
    keys = [k for k in CHANGES if k in rob["fusion_head"]]
    series = [("DeepShield v2", [rob["fusion_head"][k]["auc"] for k in keys], INK)]
    if "previous_head" in rob:
        series.append(("Previous version", [rob["previous_head"][k]["auc"] for k in keys], GREY))
    x = np.arange(len(keys))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for i, (label, values, colour) in enumerate(series):
        pos = x + (i - (len(series) - 1) / 2) * width
        ax.bar(pos, values, width=width * 0.92, color=colour, label=label)
        for xi, v in zip(pos, values):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=TEXT,
                    fontweight="bold" if i == 0 else "normal")
    ax.axhline(0.5, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(x, [CHANGES[k] for k in keys])
    ax.set_ylim(0.4, 1.06)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUC")
    ax.set_title(f"The same {rob['n']} held-out faces after each change")
    ax.legend(loc="lower left", fontsize=9.5, ncols=2, bbox_to_anchor=(0, 1.08), handlelength=1.2)
    _save(fig, "robustness.png")


def main() -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    preds = np.load(REPORTS / "test_predictions.npz", allow_pickle=True)
    accuracy_by_source(metrics)
    per_dataset(metrics)
    roc(metrics, preds)
    confusion(preds)
    robustness()
    print("Figures written to", REPORTS)


if __name__ == "__main__":
    main()
