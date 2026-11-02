"""Render evaluation figures (PNG) for the README, docs and slides.

    python ml/scripts/make_figures.py
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

# Validated categorical slots 1-3 (pass CVD checks for all pairs) and text tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#14213d", "#56657a", "#e3e8ee", "#ffffff"
DATASETS = {
    "ffhq_stylegan": "140k Faces\n(StyleGAN)",
    "dff": "DeepFakeFace\n(diffusion, swaps)",
    "celebdf": "Celeb-DF v2\n(video face swap)",
    "dfd": "Google DFD\n(video face swap)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def best_single(metrics: dict, key: str = "overall") -> str:
    singles = {k: v for k, v in metrics["test"].items() if k.startswith("zero_shot:")}
    return max(singles, key=lambda k: singles[k][key]["auc"])


def roc(metrics: dict, preds: np.lib.npyio.NpzFile) -> None:
    y = preds["y"]
    best = best_single(metrics)
    chosen = [m.split("/")[-1] for m in metrics["chosen_backbones"]]
    series = [
        ("Fused model (ours)", preds["p_fusion"], BLUE),
        (f"Best single model ({best.split(':')[1]})", preds[f"p_{best.split(':')[1]}"], ORANGE),
        ("Average of the two backbones", np.mean([preds[f"p_{c}"] for c in chosen], axis=0), AQUA),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK_2, alpha=0.6)
    for label, p, color in series:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{label}: AUC {roc_auc_score(y, p):.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (real faces flagged)",
           ylabel="True positive rate (fakes caught)")
    ax.set_title("ROC curve on the held-out test set")
    ax.legend(loc="lower right", fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(REPORTS / "roc.png", dpi=180)
    plt.close(fig)


def per_dataset(metrics: dict) -> None:
    fusion = metrics["test"]["fusion_head"]
    keys = [k for k in DATASETS if k in fusion]
    singles = {k: v for k, v in metrics["test"].items() if k.startswith("zero_shot:")}
    best_vals = [max(v[k]["auc"] for v in singles.values()) for k in keys]
    ours = [fusion[k]["auc"] for k in keys]

    fig, ax = plt.subplots(figsize=(8.4, 5))
    x = np.arange(len(keys))
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, best_vals, w, color=ORANGE, label="Best single pretrained model on that dataset")
    b2 = ax.bar(x + w / 2 + 0.01, ours, w, color=BLUE, label="Fused model (ours)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.axhline(0.5, ls="--", lw=1, color=INK_2, alpha=0.6)
    ax.text(len(keys) - 0.5, 0.51, "coin flip", ha="right", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(x, [DATASETS[k] for k in keys])
    ax.set(ylim=(0, 1.22), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ylabel="AUC (higher is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Detection quality per dataset (held-out test)")
    ax.legend(loc="upper left", ncols=2, fontsize=10, labelcolor=INK, bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(REPORTS / "per_dataset_auc.png", dpi=180)
    plt.close(fig)


def confusion(preds) -> None:
    cm = confusion_matrix(preds["y"], (preds["p_fusion"] >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(False)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#2a78d6", "#123a73"])
    ax.imshow(cm, cmap=ramp)
    for (i, j), v in np.ndenumerate(cm):
        share = v / cm[i].sum()
        ax.text(j, i, f"{v}\n{share:.0%} of row", ha="center", va="center", fontsize=11,
                color="white" if share > 0.5 else INK)
    ax.set_xticks([0, 1], ["Predicted real", "Predicted fake"])
    ax.set_yticks([0, 1], ["Actually real", "Actually fake"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion matrix (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    preds = np.load(REPORTS / "test_predictions.npz", allow_pickle=True)
    roc(metrics, preds)
    per_dataset(metrics)
    confusion(preds)
    print("Figures written to", REPORTS)


if __name__ == "__main__":
    main()
