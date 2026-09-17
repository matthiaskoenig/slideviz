"""Predict necrosis on a slide nobody has annotated, to show where to look.

The model is trained on annotated animals and applied to an unlabelled slide.
The output shows where it predicts necrosis; a person reviews the map.

    uv run python predict_unlabelled.py --cache /data/michelle/mouse/embeddings \
        --tiles /data/michelle/mouse_unlabelled --animal 158mg_m1 \
        --out /data/michelle/mouse/predictions
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

POSITIVE = 0.5
MAX_ITER = 2000
CLASS_WEIGHT = "balanced"


def main() -> None:
    """Fit on every annotated animal and predict one unlabelled slide."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="the annotated cache")
    parser.add_argument("--target-cache", type=Path, required=True,
                        help="embeddings for the unlabelled slide")
    parser.add_argument("--target-tiles", type=Path, required=True,
                        help="the unlabelled slide's tile manifest")
    parser.add_argument("--animal", required=True, help="which animal in the target")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positive", type=float, default=POSITIVE)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.cache / "embeddings_meta.json").read_text())

    train_x = np.asarray(np.load(args.cache / "embeddings.npy", mmap_mode="r"))
    necrosis = np.load(args.cache / "necrosis.npy")
    train_y = necrosis >= args.positive

    target_x = np.asarray(np.load(args.target_cache / "embeddings.npy", mmap_mode="r"))
    target_animals = np.array((args.target_cache / "animals.txt").read_text().split())
    keep = target_animals == args.animal
    if not keep.any():
        raise ValueError(f"{args.animal} is not in {args.target_cache}")

    manifest = json.loads((args.target_tiles / "manifest.json").read_text())
    rows = [r for r in manifest["tiles"] if r["animal"] == args.animal]
    if len(rows) != int(keep.sum()):
        raise ValueError(f"{len(rows)} manifest rows against {int(keep.sum())} embeddings")

    print(f"=== fit on {len(train_y):,} annotated tiles from {len(meta['animals'])} animals ===")
    scaler = StandardScaler().fit(train_x)  # every annotated animal trains this one
    head = LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)
    head.fit(scaler.transform(train_x), train_y)

    scores = head.predict_proba(scaler.transform(target_x[keep]))[:, 1]
    flagged = scores >= 0.5
    print(f"=== {args.animal}: {len(rows):,} tiles, {flagged.mean():.1%} flagged ===")

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "animal": args.animal,
        "slide": rows[0]["slide"],
        "encoder": meta["model"],
        "trained_on": meta["animals"],
        "unlabelled": True,  # nothing to score against; this map is read, not measured
        "positive_threshold": args.positive,
        "n_tiles": len(rows),
        "size_px": rows[0]["size_px"],
        "level": rows[0]["level"],
        "area_predicted": round(float(flagged.mean()), 4),
        "tiles": [
            {
                "row": r["row"],
                "col": r["col"],
                "y": r["y"],
                "x": r["x"],
                "necrosis": 0.0,  # no annotation exists, so the viewer's truth layer is empty
                "predicted": round(float(s), 4),
            }
            for r, s in zip(rows, scores, strict=True)
        ],
    }
    path = args.out / f"{args.animal}_unlabelled_predictions.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"=== wrote {path} ===")


if __name__ == "__main__":
    main()
