"""A pixel classifier for necrosis, as ilastik or QuPath train one: local colour, edge and texture measures and a random forest.

    uv run --no-sync --with scikit-learn python scripts/pixel_classifier.py --slides /data/michelle/slides \
        --tiles /data/michelle/mouse/tiles --labels <patch_labels.py output> --fields <pick_fields.py output> \
        --out <dir> --workers 40

Every pixel of the tiles' level gets 60 measures: for each colour channel the blurred intensity, the edge
strength and two texture measures at five scales from 0.44 to 7 um (skimage multiscale_basic_features). A
random forest, many small decision trees that vote, learns which measures mean necrosis. A pixel takes the
label of the encoder patch it lies in (patch_labels.py: polygons, the 30 um band, glass and lumina), which is
finer than the polygons were drawn. Training uses random tissue pixels of random training tiles of every
slide. As for the other maps, each field's forest is fitted without the field's animal, and its cutoff is
chosen on the other 19 by the tile rule (each left out in turn, lowest mean area error).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from slideviz.io.reader import open_slide

SIGMA_PX = (1, 16)  # 0.44 to 7 um at the tiles' level
MARGIN = 64  # px around every crop, four times the coarsest blur, so edge pixels see their surroundings
TILES_PER_SLIDE = 300
PIXELS_PER_TILE = 40
TISSUE = 0.5  # patch_labels.py's HALF
SEED = 0
THREADS = 4
CUTOFFS = np.round(np.arange(0.30, 0.91, 0.025), 3)  # train_necrosis.CUTOFFS, the tile head's grid
FOREST = {"n_estimators": 100, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced",
          "random_state": SEED}
FIELD_PAD = 40  # px kept around each field, for a report that filters past the field's border


def measures(rgb: np.ndarray) -> np.ndarray:
    """The 60 measures of every pixel of an RGB crop, as float32."""
    return multiscale_basic_features(rgb, channel_axis=-1, sigma_min=SIGMA_PX[0],
                                     sigma_max=SIGMA_PX[1]).astype(np.float32)


def read_crop(level: object, top: int, left: int, size: int) -> np.ndarray:
    """A square of a pyramid level with MARGIN around it, mirrored where it passes the slide's edge."""
    height, width = level.shape[:2]
    y0, x0 = max(0, top - MARGIN), max(0, left - MARGIN)
    y1, x1 = min(height, top + size + MARGIN), min(width, left + size + MARGIN)
    crop = np.asarray(level[y0:y1, x0:x1])[..., :3]
    pads = ((y0 - (top - MARGIN), top + size + MARGIN - y1), (x0 - (left - MARGIN), left + size + MARGIN - x1), (0, 0))
    return np.pad(crop, pads, mode="reflect")


def pixel_patch(size_px: int, grid: int) -> np.ndarray:
    """The patch row (or column) each tile pixel's centre falls in, with patch edges at r * size_px / grid."""
    return ((np.arange(size_px) + 0.5) * grid / size_px).astype(int)


def sample_slide(job: tuple) -> dict:
    """Random tissue pixels of random training tiles of one slide: their measures, labels and weights."""
    slide, index, rows, labels, tissue, slides, seed = job
    rng = np.random.default_rng(seed)
    _, levels = open_slide(slides / f"{slide}.ome.tiff")
    picked = rng.choice(len(rows), min(TILES_PER_SLIDE, len(rows)), replace=False)
    x, label, weight, tile = [], [], [], []
    for k in picked:
        row = rows[k]
        size = row["size_px"]
        to_patch = pixel_patch(size, labels.shape[1])
        pixel_label = labels[k][np.ix_(to_patch, to_patch)]
        candidates = np.flatnonzero((tissue[k][np.ix_(to_patch, to_patch)] >= TISSUE).ravel())
        if not candidates.size:
            continue
        chosen = rng.choice(candidates, min(PIXELS_PER_TILE, candidates.size), replace=False)
        found = measures(read_crop(levels[row["level"]], row["y"], row["x"], size))[MARGIN:MARGIN + size,
                                                                                       MARGIN:MARGIN + size]
        x.append(found.reshape(-1, found.shape[-1])[chosen])
        label.append(pixel_label.ravel()[chosen])
        # each sampled pixel stands for its tile's tissue pixels, so a sampled area is an area of pixels
        weight.append(np.full(chosen.size, candidates.size / chosen.size, np.float32))
        tile.append(np.full(chosen.size, index[k]))
    return {"x": np.concatenate(x), "label": np.concatenate(label), "weight": np.concatenate(weight),
            "tile": np.concatenate(tile)}


def area_error(scores: np.ndarray, truth: np.ndarray, cutoff: float, weight: np.ndarray) -> float:
    """train_necrosis.area_error on a weighted sample: how far the called area lies from the drawn one."""
    return abs(float(np.average(scores >= cutoff, weights=weight) - np.average(truth, weights=weight)))


def forest(sample: dict, held: tuple, threads: int) -> RandomForestClassifier:
    """A forest on every labelled sample pixel outside the held animals."""
    train = ~np.isin(sample["animal"], held) & (sample["label"] >= 0)
    model = RandomForestClassifier(**FOREST, n_jobs=threads)
    with threadpool_limits(threads):
        model.fit(sample["x"][train], sample["label"][train] == 1)
    return model


def fit_pair(job: tuple) -> dict:
    """One forest without two animals, and its area error on each of them at every cutoff."""
    sample_path, held, threads = job
    sample = dict(np.load(sample_path))
    model = forest(sample, held, threads)
    errors = {}
    for animal in held:
        own = sample["animal"] == animal
        scores = model.predict_proba(sample["x"][own])[:, 1]
        errors[animal] = [area_error(scores, sample["label"][own] == 1, c, sample["weight"][own]) for c in CUTOFFS]
    return {"held": list(held), "errors": errors}


def fit_field(job: tuple) -> dict:
    """The forest without one animal: its held-out sample scores and its score map over the animal's field."""
    sample_path, animal, cutoff, field, level, slides, out, threads = job
    sample = dict(np.load(sample_path))
    model = forest(sample, (animal,), threads)
    own = sample["animal"] == animal
    scores = model.predict_proba(sample["x"][own])[:, 1]
    truth, weight, labelled = sample["label"][own] == 1, sample["weight"][own], sample["label"][own] >= 0
    called = scores >= cutoff
    both, drawn, kept = (int(v.sum()) for v in (truth & called & labelled, truth & labelled, called & labelled))

    info, levels = open_slide(slides / f"{field['slide']}.ome.tiff")
    factor = 2 ** round(np.log2(levels[0].shape[1] / levels[level].shape[1]))
    um = info.pixel_size_um * factor
    half = round(field["size_um"] / 2 / um) + FIELD_PAD
    top, left = (round(v / factor) - half for v in field["centre_level0_rc"])
    found = measures(read_crop(levels[level], top, left, 2 * half))[MARGIN:MARGIN + 2 * half,
                                                                    MARGIN:MARGIN + 2 * half]
    with threadpool_limits(threads):
        score = model.predict_proba(found.reshape(-1, found.shape[-1]))[:, 1].reshape(2 * half, 2 * half)
    meta = {"animal": animal, "slide": field["slide"], "cutoff": cutoff, "level": level,
            "origin_level_rc": [top, left], "um_per_px": um, "field": {"centre_level0_rc": field["centre_level0_rc"]}}
    np.savez_compressed(out / f"{animal}_pixel_field.npz", score=score.astype(np.float32), meta=json.dumps(meta))
    return {
        "held_out": animal,
        "cutoff": cutoff,
        "n_train_pixels": int((~np.isin(sample["animal"], (animal,)) & (sample["label"] >= 0)).sum()),
        "area_true": round(float(np.average(truth, weights=weight)), 4),
        "area_predicted": round(float(np.average(called, weights=weight)), 4),
        "dsc": round(2 * both / (drawn + kept), 4) if drawn + kept else None,
        "iou": round(both / (drawn + kept - both), 4) if drawn + kept else None,
        "field_called_share": round(float((score >= cutoff).mean()), 4),
    }


def code_version() -> str:
    """The git commit the script ran from, marked dirty if the tree had changes and untracked if it is not in it."""
    here = Path(__file__).parent
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=here)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", Path(__file__).name],
                             capture_output=True, check=False, cwd=here).returncode == 0
    version = found.stdout.strip()
    return "unknown" if not version else version if tracked else f"{version}-untracked"


def main() -> None:
    """Sample pixels, choose every animal's cutoff, fit its forest, and write its field's score map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="H&E OME-TIFFs")
    parser.add_argument("--tiles", type=Path, required=True, help="export_tiles.py output the labels belong to")
    parser.add_argument("--labels", type=Path, required=True, help="patch_labels.py output")
    parser.add_argument("--fields", type=Path, required=True, help="pick_fields.py output")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")

    manifest = json.loads((args.tiles / "manifest.json").read_text())
    rows = manifest["tiles"]
    if (args.labels / "patch_labels_tiles.txt").read_text().split() != [r["tile"] for r in rows]:
        raise SystemExit("the labels list other tiles than the manifest")
    labels = np.load(args.labels / "patch_labels.npy", mmap_mode="r")
    tissue = np.load(args.labels / "patch_tissue.npy", mmap_mode="r")
    chosen = json.loads(args.fields.read_text())
    names = list(manifest["animals"])
    if missing := set(chosen["fields"]) - set(names):
        raise SystemExit(f"fields without training tiles: {sorted(missing)}")

    by_slide: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_slide.setdefault(row["slide"], []).append(i)
    jobs = [(slide, np.array(index), [rows[i] for i in index], np.asarray(labels[index]), np.asarray(tissue[index]),
             args.slides, [SEED, k]) for k, (slide, index) in enumerate(sorted(by_slide.items()))]

    args.out.mkdir(parents=True, exist_ok=True)
    context = get_context("spawn")  # fresh workers, so no BLAS thread state is copied into them
    with ProcessPoolExecutor(max(1, min(args.workers, len(jobs))), mp_context=context) as pool:
        parts = list(pool.map(sample_slide, jobs))
    sample = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    sample["animal"] = np.array([rows[i]["animal"] for i in sample["tile"]])
    sample_path = args.out / "sample.npz"
    np.savez(sample_path, **sample)
    print(f"=== {len(sample['label']):,} pixels sampled from {len(jobs)} slides ===", flush=True)

    # the forest without animals a and b checks the cutoff for a held out (on b) and for b held out (on a)
    inner: dict[str, dict[str, list[float]]] = {a: {} for a in names}
    with ProcessPoolExecutor(max(1, args.workers), mp_context=context) as pool:
        for result in pool.map(fit_pair, [(sample_path, pair, args.threads) for pair in combinations(names, 2)]):
            a, b = result["held"]
            inner[a][b], inner[b][a] = result["errors"][b], result["errors"][a]
    traces = {a: np.mean([inner[a][b] for b in names if b != a], axis=0) for a in names}
    cutoffs = {a: float(CUTOFFS[int(np.argmin(traces[a]))]) for a in names}
    print(f"=== cutoffs chosen: {sorted(set(cutoffs.values()))} ===", flush=True)

    (level,) = {r["level"] for r in rows}
    field_jobs = [(sample_path, a, cutoffs[a], {**f, "size_um": chosen["size_um"]}, level, args.slides, args.out,
                   args.threads) for a, f in chosen["fields"].items()]
    with ProcessPoolExecutor(max(1, min(args.workers, len(field_jobs))), mp_context=context) as pool:
        folds = list(pool.map(fit_field, field_jobs))
    for f in folds:
        f["cutoff_trace"] = [{"cutoff": float(c), "mean_absolute_area_error": round(float(m), 4)}
                             for c, m in zip(CUTOFFS, traces[f["held_out"]], strict=True)]
        print(f"{f['held_out']}: cutoff {f['cutoff']}, area {f['area_true']:.1%} drawn, {f['area_predicted']:.1%} "
              f"called on its sample pixels, DSC {f['dsc']}, field {f['field_called_share']:.1%}", flush=True)

    necrotic = [f for f in folds if f["area_true"] > 0]
    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "measures": (f"skimage multiscale_basic_features on RGB at the tiles' level, sigma {SIGMA_PX[0]} to "
                     f"{SIGMA_PX[1]} px (intensity, edges, texture), {MARGIN} px margin"),
        "forest": FOREST,
        "sample": {"tiles_per_slide": TILES_PER_SLIDE, "pixels_per_tile": PIXELS_PER_TILE, "seed": SEED,
                   "n_pixels": len(sample["label"]),
                   "rule": f"random pixels of patches with at least {TISSUE} tissue; trained on those labelled 0 or 1"},
        "split": "leave one animal out; the cutoff chosen on the 19 training animals, each left out in turn",
        "cutoff_objective": "mean absolute area error over the inner animals, on their weighted sample pixels",
        "cutoff_grid": [float(c) for c in CUTOFFS],
        "metric_caveat": ("area, DSC and IoU on the held-out animal's sample pixels against labels from polygons "
                          "that stop short of the necrosis edge; DSC and IoU leave the 30 um band out; information"),
        "mean_absolute_area_error": round(float(np.mean([abs(f["area_predicted"] - f["area_true"])
                                                         for f in necrotic])), 4) if necrotic else None,
        "mean_dsc_necrotic": round(float(np.mean([f["dsc"] for f in necrotic])), 4) if necrotic else None,
        "folds": folds,
        "files": {"<animal>_pixel_field.npz": "score (float32) over the field plus a margin on level 1, and meta",
                  "sample.npz": "the sampled pixels: measures, label, weight, tile, animal"},
        "inputs": {"tiles": str(args.tiles), "tiles_written": manifest["written"], "labels": str(args.labels),
                   "fields": str(args.fields), "fields_written": chosen["written"], "slides": str(args.slides)},
    }
    (args.out / "pixel_classifier.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"=== {len(folds)} fields written to {args.out} ===")


if __name__ == "__main__":
    main()
