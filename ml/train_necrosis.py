"""Train a linear head on cached tile embeddings, holding out one animal per fold.

Animals correspond to slides, so random tile splits would leak slide-specific features.

    uv run python train_necrosis.py --cache /data/michelle/mouse/embeddings
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# a tile is called necrotic above this much coverage; tiles between the two are ambiguous
POSITIVE = 0.5

MAX_ITER = 2000
CLASS_WEIGHT = "balanced"


def load_cache(cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Embeddings, necrosis fractions, animals, tile paths and their provenance."""
    meta = json.loads((cache / "embeddings_meta.json").read_text())
    embeddings = np.asarray(np.load(cache / "embeddings.npy", mmap_mode="r"))
    necrosis = np.load(cache / "necrosis.npy")
    animals = np.array((cache / "animals.txt").read_text().split())
    tiles = (cache / "tiles.txt").read_text().split()
    if not len(embeddings) == len(necrosis) == len(animals) == len(tiles):
        raise ValueError(
            f"cache is inconsistent: {len(embeddings)} embeddings, {len(necrosis)} "
            f"labels, {len(animals)} animals, {len(tiles)} tiles"
        )
    return embeddings, necrosis, animals, tiles, meta


def fold(
    embeddings: np.ndarray,
    target: np.ndarray,
    animals: np.ndarray,
    held_out: str,
    keep: np.ndarray,
) -> dict:
    """Fit on every other animal and score on this one."""
    test = (animals == held_out) & keep
    train = (animals != held_out) & keep

    scaler = StandardScaler().fit(embeddings[train])  # train alone, so test statistics stay out
    head = LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)
    head.fit(scaler.transform(embeddings[train]), target[train])

    scores = head.predict_proba(scaler.transform(embeddings[test]))[:, 1]
    predicted = scores >= 0.5
    truth = target[test]

    matrix = confusion_matrix(truth, predicted, labels=[False, True])
    single_class = len(np.unique(truth)) < 2  # AUROC is undefined on one class
    return {
        "held_out": held_out,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "test_positive": int(truth.sum()),
        "test_positive_rate": round(float(truth.mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 4),
        "f1": round(float(f1_score(truth, predicted, zero_division=0)), 4),
        "auroc": None if single_class else round(float(roc_auc_score(truth, scores)), 4),
        "average_precision": None
        if single_class
        else round(float(average_precision_score(truth, scores)), 4),
        "confusion": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        # the readout is percent necrotic area, so slide level is what the project reports
        "area_true": round(float(truth.mean()), 4),
        "area_predicted": round(float(predicted.mean()), 4),
    }


def main() -> None:
    """Run one fold per animal and write the metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="embed_tiles.py output")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--positive", type=float, default=POSITIVE,
                        help="coverage above which a tile counts as necrotic")
    parser.add_argument("--drop-boundary", action="store_true",
                        help="train and test only on tiles below 0.1 or above 0.9")
    parser.add_argument("--tag", default="", help="names this run in the output file")
    args = parser.parse_args()

    out = args.out or args.cache
    out.mkdir(parents=True, exist_ok=True)

    embeddings, necrosis, animals, _, meta = load_cache(args.cache)
    target = necrosis >= args.positive
    keep = (
        (necrosis <= 0.1) | (necrosis >= 0.9) if args.drop_boundary
        else np.ones(len(necrosis), dtype=bool)
    )

    print(f"=== cache: {embeddings.shape[0]:,} x {embeddings.shape[1]} from {meta['model']} ===")
    for animal in meta["animals"]:
        rows = animals == animal
        print(f"  {animal:<12} {rows.sum():>6,} tiles, {target[rows].mean():>6.1%} necrotic")
    if args.drop_boundary:
        print(f"  boundary filter: {(~keep).sum():,} of {len(keep):,} tiles dropped")

    print(f"\n=== leave one animal out, positive at coverage >= {args.positive} ===")
    folds = [fold(embeddings, target, animals, a, keep) for a in meta["animals"]]

    header = f"{'held out':<12}{'train':>8}{'test':>7}{'bal acc':>9}{'F1':>8}{'AUROC':>8}{'AP':>8}"
    print(header)
    for f in folds:
        auroc = "n/a" if f["auroc"] is None else f"{f['auroc']:.4f}"
        ap = "n/a" if f["average_precision"] is None else f"{f['average_precision']:.4f}"
        print(
            f"{f['held_out']:<12}{f['n_train']:>8,}{f['n_test']:>7,}"
            f"{f['balanced_accuracy']:>9.4f}{f['f1']:>8.4f}{auroc:>8}{ap:>8}"
        )

    mean_balanced = float(np.mean([f["balanced_accuracy"] for f in folds]))
    mean_f1 = float(np.mean([f["f1"] for f in folds]))
    print(f"{'mean':<12}{'':>8}{'':>7}{mean_balanced:>9.4f}{mean_f1:>8.4f}")

    # Exclude control slides from the average; they have no positive tiles.
    scored = [f for f in folds if f["test_positive"]]
    if len(scored) != len(folds):
        mean_balanced_scored = float(np.mean([f["balanced_accuracy"] for f in scored]))
        mean_f1_scored = float(np.mean([f["f1"] for f in scored]))
        print(
            f"{'mean, positive folds only':<12}{'':>8}"
            f"{mean_balanced_scored:>16.4f}{mean_f1_scored:>8.4f}"
            f"   ({len(scored)} of {len(folds)})"
        )

    print("\n=== necrotic area per slide, the project's actual readout ===")
    print(f"{'animal':<12}{'annotated':>11}{'predicted':>11}{'error':>9}")
    for f in folds:
        error = f["area_predicted"] - f["area_true"]
        print(
            f"{f['held_out']:<12}{f['area_true']:>10.1%}{f['area_predicted']:>11.1%}"
            f"{error:>+9.1%}"
        )

    results = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "tag": args.tag,
        "cache": str(args.cache),
        "encoder": meta["model"],
        "embedding_dim": meta["embedding_dim"],
        "head": "LogisticRegression on standardised embeddings",
        "class_weight": CLASS_WEIGHT,
        "positive_threshold": args.positive,
        "boundary_dropped": bool(args.drop_boundary),
        "split": "leave one animal out",
        "folds": folds,
        "mean_balanced_accuracy": round(mean_balanced, 4),
        "mean_f1": round(mean_f1, 4),
        "n_folds_with_positives": len([f for f in folds if f["test_positive"]]),
        "mean_balanced_accuracy_positive_folds": round(
            float(np.mean([f["balanced_accuracy"] for f in folds if f["test_positive"]])), 4
        ),
        "mean_f1_positive_folds": round(
            float(np.mean([f["f1"] for f in folds if f["test_positive"]])), 4
        ),
        "n_animals": len(meta["animals"]),
        # derived to avoid stale hardcoded counts
        "caveat": (
            f"{len(meta['animals'])} animals, so each fold trains on "
            f"{len(meta['animals']) - 1}; a fold is one slide, not a cohort"
        ),
        # identify matched embeddings
        "stain_matched": meta.get("stain_matched", False),
        "stain_reference": meta.get("stain_reference"),
    }
    # keep tagged runs from overwriting each other
    suffix = "_no_boundary" if args.drop_boundary else ""
    tag = f"_{args.tag}" if args.tag else ""
    name = f"necrosis_results{tag}{suffix}.json"
    (out / name).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n=== wrote {out / name} ===")


if __name__ == "__main__":
    main()
