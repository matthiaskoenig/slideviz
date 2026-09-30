"""HepatoBench figures: per-class curves, confusion matrix, comparison with the paper, patch grid.

    uv run --with matplotlib python scripts/hepatobench_figures.py \
        --results <linear_head_results.json> --scores <test_scores.npz> \
        --zips <hepatobench/zips> --out <figures>
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

# Ling et al. 2026 (arXiv 2604.22858), Fig. 2a: frozen encoder plus head, slide-level split
PAPER = {
    "UNI": [0.913, 0.929, 0.913, 0.920, 0.996, 0.967],
    "GigaPath": [0.967, 0.971, 0.967, 0.969, 0.999, 0.990],
    "Virchow-V2": [0.968, 0.969, 0.968, 0.969, 1.000, 0.993],
}
METRICS = ["BACC", "Precision", "Recall", "F1", "AUC", "AUPR"]
OURS = ["balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
        "macro_auc", "macro_aupr"]

CORRECT_PER_CLASS = 5
WRONG_PER_CLASS = 4
COLOURS = plt.get_cmap("tab10").colors


def short(name: str) -> str:
    """01_TUM -> TUM."""
    return name.split("_", 1)[1]


def curves(results: dict, scores: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """One-vs-rest ROC and precision-recall curve per class."""
    classes = results["classes"]
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 5))
    for index, name in enumerate(classes):
        truth = labels == index
        fpr, tpr, _ = roc_curve(truth, scores[:, index])
        precision, recall, _ = precision_recall_curve(truth, scores[:, index])
        colour = COLOURS[index]
        roc_ax.plot(fpr, tpr, color=colour, lw=1.2,
                    label=f"{short(name)}  AUC {results['per_class_auc'][name]:.4f}")
        pr_ax.plot(recall, precision, color=colour, lw=1.2,
                   label=f"{short(name)}  AP {results['per_class_ap'][name]:.4f}")
    roc_ax.set(xlabel="false positive rate", ylabel="true positive rate", xscale="log",
               xlim=(1e-4, 1), ylim=(0.8, 1.005), title="ROC, one class against the rest")
    pr_ax.set(xlabel="recall", ylabel="precision", xlim=(0.8, 1.005), ylim=(0.8, 1.005),
              title="Precision-recall, one class against the rest")
    for ax in (roc_ax, pr_ax):
        ax.legend(fontsize=8, loc="lower left" if ax is pr_ax else "lower right")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Frozen Lunit ViT-S/8 + linear head, {results['n_test']:,} test patches "
                 "(random split)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion(results: dict, path: Path) -> None:
    """Confusion matrix as counts, shaded by the share of each true class."""
    matrix = np.array(results["confusion_matrix"])
    names = [short(c) for c in results["classes"]]
    share = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for r in range(len(names)):
        for c in range(len(names)):
            ax.text(c, r, f"{matrix[r, c]:,}", ha="center", va="center", fontsize=8,
                    color="white" if share[r, c] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="predicted", ylabel="true", title="Confusion matrix, test patches")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def comparison(results: dict, path: Path) -> None:
    """The paper's six metrics for its three encoders beside this run's."""
    models = {**PAPER, "Lunit ViT-S/8 (ours)": [results[k] for k in OURS]}
    width = 0.2
    x = np.arange(len(METRICS))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for index, (model, values) in enumerate(models.items()):
        ours = index == len(models) - 1
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=model,
                      color="#c0392b" if ours else plt.get_cmap("Blues")(0.35 + 0.2 * index),
                      hatch="//" if ours else None, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1)
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("score, macro over 7 classes")
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    ax.set_title("HepatoBench: paper (slide-level split) against this run (random split, "
                 "inflated)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pick(scores: np.ndarray, labels: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Per class, the most confident correct test patches and the most confident errors."""
    predicted = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    chosen = []
    for index in range(scores.shape[1]):
        right = np.flatnonzero((labels == index) & (predicted == index))
        wrong = np.flatnonzero((labels == index) & (predicted != index))
        chosen.append((list(right[np.argsort(-confidence[right])][:CORRECT_PER_CLASS]),
                       list(wrong[np.argsort(-confidence[wrong])][:WRONG_PER_CLASS])))
    return chosen


def read_patch(zips: Path, name: str) -> np.ndarray:
    """One patch image out of its class zip."""
    with zipfile.ZipFile(zips / f"{name.rsplit('_', 1)[0]}.zip") as archive:
        return np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("RGB"))


def grid(results: dict, scores: np.ndarray, labels: np.ndarray, names: np.ndarray,
         zips: Path, path: Path) -> None:
    """Per class, confident correct patches, then the most confident errors."""
    classes = results["classes"]
    predicted = scores.argmax(axis=1)
    columns = CORRECT_PER_CLASS + WRONG_PER_CLASS
    fig, axes = plt.subplots(len(classes), columns, figsize=(columns * 1.35, len(classes) * 1.5))
    for row, (right, wrong) in enumerate(pick(scores, labels)):
        for col in range(columns):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            index = (right + [None] * CORRECT_PER_CLASS)[col] if col < CORRECT_PER_CLASS \
                else (wrong + [None] * WRONG_PER_CLASS)[col - CORRECT_PER_CLASS]
            if index is None:
                ax.axis("off")
                continue
            ax.imshow(read_patch(zips, str(names[index])))
            error = col >= CORRECT_PER_CLASS
            ax.set_title(f"as {short(classes[predicted[index]])} {scores[index].max():.2f}"
                         if error else f"{scores[index].max():.2f}", fontsize=6.5,
                         color="#c0392b" if error else "black")
            for spine in ax.spines.values():
                spine.set_edgecolor("#c0392b" if error else "black")
                spine.set_linewidth(1.5 if error else 0.5)
        axes[row, 0].set_ylabel(short(classes[row]), fontsize=9)
    fig.suptitle("Test patches per true class: most confident correct (left), most confident "
                 "errors in red (right)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Write the four figures."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--zips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    stored = np.load(args.scores)
    scores, labels, names = stored["scores"], stored["labels"], stored["names"]
    args.out.mkdir(parents=True, exist_ok=True)

    curves(results, scores, labels, args.out / "curves.png")
    confusion(results, args.out / "confusion.png")
    comparison(results, args.out / "comparison.png")
    grid(results, scores, labels, names, args.zips, args.out / "patch_grid.png")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
