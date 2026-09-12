"""Train a linear head on cached embeddings and report per-class results.

The split is random: HepatoBench filenames carry a class and an index only, so
patches from one slide land on both sides and the score is inflated.

    uv run python train_linear_head.py --cache /data/michelle/hepatobench/embeddings
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TEST_FRACTION = 0.2
SEED = 0  # fixed, so a rerun reproduces the split
MAX_ITER = 2000

# REA is 1,540 against FIB's 26,364, so weight by inverse frequency
CLASS_WEIGHT = "balanced"


def load_cache(cache: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Embeddings, labels, patch filenames and the provenance written with them."""
    meta = json.loads((cache / "embeddings_meta.json").read_text())
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy")
    names = (cache / "patches.txt").read_text().split()
    if not len(embeddings) == len(labels) == len(names):
        raise ValueError(
            f"cache is inconsistent: {len(embeddings)} embeddings, "
            f"{len(labels)} labels, {len(names)} names"
        )
    return np.asarray(embeddings), labels, names, meta


def main() -> None:
    """Fit the head on cached embeddings and write metrics and errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="extract_embeddings.py output")
    parser.add_argument("--out", type=Path, default=None, help="where to write results")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-class-weight", action="store_true",
                        help="drop inverse-frequency weighting, to see what the imbalance costs")
    args = parser.parse_args()

    out = args.out or args.cache
    out.mkdir(parents=True, exist_ok=True)

    embeddings, labels, names, meta = load_cache(args.cache)
    classes = meta["classes"]
    print(f"=== cache: {embeddings.shape[0]:,} x {embeddings.shape[1]} from {meta['model']} ===")
    for i, name in enumerate(classes):
        print(f"  {name}  {int((labels == i).sum()):>7,}")

    if not meta.get("patient_ids_available", False):
        print("\n  NOTE: no slide id in this dataset, so the split is random and inflated.")

    # stratify keeps the 1,540-patch class at full proportion in both halves
    train_x, test_x, train_y, test_y, _, test_names = train_test_split(
        embeddings, labels, names,
        test_size=args.test_fraction, random_state=args.seed, stratify=labels,
    )
    print(f"\n=== split: {len(train_y):,} train, {len(test_y):,} test (seed {args.seed}) ===")

    scaler = StandardScaler().fit(train_x)  # train alone, so test statistics stay out

    weight = None if args.no_class_weight else CLASS_WEIGHT
    print(f"=== fitting linear head (class_weight={weight}) ===")
    head = LogisticRegression(max_iter=MAX_ITER, class_weight=weight)
    head.fit(scaler.transform(train_x), train_y)

    predicted = head.predict(scaler.transform(test_x))
    accuracy = accuracy_score(test_y, predicted)
    balanced = balanced_accuracy_score(test_y, predicted)  # rare classes count equally
    macro_f1 = f1_score(test_y, predicted, average="macro")

    print(f"\n=== results on {len(test_y):,} held-out patches ===")
    print(f"  accuracy           {accuracy:.4f}")
    print(f"  balanced accuracy  {balanced:.4f}")
    print(f"  macro F1           {macro_f1:.4f}")
    print()
    print(classification_report(test_y, predicted, target_names=classes, digits=4, zero_division=0))

    matrix = confusion_matrix(test_y, predicted)
    print("=== confusion matrix: rows are true, columns are predicted ===")
    print(f"  {'':<8}" + "".join(f"{c[3:]:>8}" for c in classes))
    for i, row in enumerate(matrix):
        print(f"  {classes[i]:<8}" + "".join(f"{v:>8,}" for v in row) + f"   ({row.sum():,})")

    print("\n=== most confused pairs ===")
    confusions = [
        (matrix[i, j], classes[i], classes[j], matrix[i, j] / matrix[i].sum())
        for i in range(len(classes)) for j in range(len(classes)) if i != j and matrix[i, j]
    ]
    for count, true, wrong, share in sorted(confusions, reverse=True)[:5]:
        print(f"  {true} read as {wrong}: {count:,} ({share:.1%} of {true})")

    # every error keeps its filename, so it can be pulled up as an image
    wrong_rows = np.flatnonzero(predicted != test_y)
    errors = [
        {"patch": test_names[i], "true": classes[test_y[i]], "predicted": classes[predicted[i]]}
        for i in wrong_rows
    ]
    (out / "misclassified.json").write_text(json.dumps(errors, indent=2) + "\n")

    results = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "cache": str(args.cache),
        "encoder": meta["model"],
        "embedding_dim": meta["embedding_dim"],
        "head": "LogisticRegression on standardised embeddings",
        "class_weight": weight,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": {
            c: round(float(v), 4)
            for c, v in zip(classes, f1_score(test_y, predicted, average=None, zero_division=0))
        },
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
        "split_by_slide": meta.get("patient_ids_available", False),
        "caveat": "random split: patches from one slide appear in train and test",
    }
    (out / "linear_head_results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n=== wrote {out}/linear_head_results.json ===")
    print(f"  {len(errors):,} misclassified patches listed in misclassified.json")


if __name__ == "__main__":
    main()
