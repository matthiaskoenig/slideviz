"""Predict one held-out slide's tiles and write the scores next to their positions.

The model that predicts a slide is never trained on it, so this runs one fold of the
leave-one-animal-out split and writes only that animal. The output carries each tile's
grid position, so the viewer can paint the scores back onto the slide.

    uv run python predict_slide.py --cache /data/michelle/mouse/embeddings \
        --animal 281mg_m1 --out /data/michelle/mouse/predictions
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
    """Fit on every other animal, predict this one, and write the scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="embed_tiles.py output")
    parser.add_argument("--tiles", type=Path, required=True, help="export_tiles.py output")
    parser.add_argument("--animal", required=True, help="the animal to hold out and predict")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positive", type=float, default=POSITIVE)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.cache / "embeddings_meta.json").read_text())
    manifest = json.loads((args.tiles / "manifest.json").read_text())

    embeddings = np.asarray(np.load(args.cache / "embeddings.npy", mmap_mode="r"))
    necrosis = np.load(args.cache / "necrosis.npy")
    animals = np.array((args.cache / "animals.txt").read_text().split())
    target = necrosis >= args.positive

    if args.animal not in set(animals):
        raise ValueError(f"{args.animal} is not in the cache: {sorted(set(animals))}")

    test, train = animals == args.animal, animals != args.animal
    print(f"=== fit on {train.sum():,} tiles, predict {test.sum():,} of {args.animal} ===")

    scaler = StandardScaler().fit(embeddings[train])  # train alone, so test statistics stay out
    head = LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)
    head.fit(scaler.transform(embeddings[train]), target[train])
    scores = head.predict_proba(scaler.transform(embeddings[test]))[:, 1]

    rows = [r for r in manifest["tiles"] if r["animal"] == args.animal]
    if len(rows) != len(scores):
        raise ValueError(f"{len(rows)} manifest rows against {len(scores)} predictions")

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "animal": args.animal,
        "slide": rows[0]["slide"],
        "encoder": meta["model"],
        "trained_on": sorted(set(animals[train])),
        "positive_threshold": args.positive,
        "n_tiles": len(rows),
        "size_px": rows[0]["size_px"],
        "level": rows[0]["level"],
        "area_annotated": round(float(target[test].mean()), 4),
        "area_predicted": round(float((scores >= 0.5).mean()), 4),
        "tiles": [
            {
                "row": r["row"],
                "col": r["col"],
                "y": r["y"],
                "x": r["x"],
                "necrosis": r["necrosis"],
                "predicted": round(float(s), 4),
            }
            for r, s in zip(rows, scores, strict=True)
        ],
    }
    path = args.out / f"{args.animal}_predictions.json"
    path.write_text(json.dumps(record, indent=2) + "\n")

    print(f"  annotated {record['area_annotated']:.1%}, predicted {record['area_predicted']:.1%}")
    print(f"=== wrote {path} ===")


if __name__ == "__main__":
    main()
