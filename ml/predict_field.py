"""Score overlapping field tiles with each animal's held-out necrosis head and nested cutoff.

    uv run python predict_field.py --cache /data/michelle/mouse/embeddings \
        --nested <necrosis_results_nested_area.json> --field-cache <field embeddings> \
        --field-tiles <field tiles> --out <dir>

The head for an animal is fitted on every other animal, as in predict_slide.py, and must
reproduce the nested run's area on the animal's own tiles before its field is scored.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from predict_slide import CLASS_WEIGHT, MAX_ITER, POSITIVE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Fit each field animal's held-out head, check it, and write its field scores."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="the annotated 75 um cache")
    parser.add_argument("--nested", type=Path, required=True,
                        help="train_necrosis.py --nested results, for cutoffs and areas")
    parser.add_argument("--field-cache", type=Path, required=True, help="embedded field tiles")
    parser.add_argument("--field-tiles", type=Path, required=True,
                        help="export_field_tiles.py output")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    folds = {f["held_out"]: f for f in json.loads(args.nested.read_text())["folds"]}
    meta = json.loads((args.cache / "embeddings_meta.json").read_text())
    embeddings = np.asarray(np.load(args.cache / "embeddings.npy", mmap_mode="r"))
    target = np.load(args.cache / "necrosis.npy") >= POSITIVE
    animals = np.array((args.cache / "animals.txt").read_text().split())

    manifest = json.loads((args.field_tiles / "manifest.json").read_text())
    field_x = np.asarray(np.load(args.field_cache / "embeddings.npy", mmap_mode="r"))
    field_animals = np.array((args.field_cache / "animals.txt").read_text().split())

    for animal, geometry in manifest["fields"].items():
        fold = folds[animal]
        train, own = animals != animal, animals == animal
        scaler = StandardScaler().fit(embeddings[train])
        head = LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)
        head.fit(scaler.transform(embeddings[train]), target[train])

        # the same model as the manuscript numbers, or nothing is written
        own_area = round(float((head.predict_proba(scaler.transform(embeddings[own]))[:, 1]
                                >= fold["cutoff"]).mean()), 4)
        if own_area != fold["area_predicted"]:
            raise SystemExit(f"{animal}: area {own_area} does not reproduce the nested "
                             f"run's {fold['area_predicted']}, nothing written")

        keep = field_animals == animal
        rows = [r for r in manifest["tiles"] if r["animal"] == animal]
        if len(rows) != int(keep.sum()):
            raise ValueError(f"{animal}: {len(rows)} manifest rows against {int(keep.sum())} embeddings")
        scores = head.predict_proba(scaler.transform(field_x[keep]))[:, 1]

        record = {
            "written": datetime.now(UTC).isoformat(timespec="seconds"),
            "animal": animal,
            "slide": geometry["slide"],
            "encoder": meta["model"],
            "trained_on": sorted(set(animals[train])),
            "cutoff": fold["cutoff"],
            "cutoff_source": str(args.nested),
            "area_reproduced": own_area,
            "field": geometry,
            "tiles": [{"row": r["row"], "col": r["col"], "predicted": round(float(s), 4)}
                      for r, s in zip(rows, scores, strict=True)],
        }
        path = args.out / f"{animal}_field.json"
        path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"{animal}: area {own_area} reproduced, {len(rows):,} field tiles, "
              f"{(scores >= fold['cutoff']).mean():.1%} above cutoff {fold['cutoff']}", flush=True)


if __name__ == "__main__":
    main()
