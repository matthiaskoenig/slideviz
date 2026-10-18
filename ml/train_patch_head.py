"""Train the per-patch necrosis head: the tile head's classifier, fitted on the encoder's patch descriptions.

    uv run python train_patch_head.py --embeddings <embed_tiles.py --patches output> \
        --labels <patch_labels.py output> --out <dir> --workers 40

The encoder describes every 2.7 um patch of a tile (patches.npy), and the head is train_necrosis.py's
logistic regression fitted on those descriptions with patch_labels.py's labels, so it calls necrosis
per patch instead of per tile. Each animal gets a head fitted on the other 19, with its cutoff chosen
on those 19 by the tile head's rule (each one left out in turn, lowest mean area error), so no animal
informs its own map. Training uses a fixed random sample of --per-tile tissue patches per tile; then
every patch of every training tile is scored by the head that never saw its animal.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_necrosis import CLASS_WEIGHT, CUTOFFS, MAX_ITER

# a tile's patches overlap in what they see, so a few per tile carry most of what the tile can teach
PER_TILE = 16
SEED = 0
TISSUE = 0.5  # patch_labels.py's HALF: a patch at least half tissue
THREADS = 4  # per worker, so 40 workers stay inside gpu1's 256 cores
CHUNK = 256  # tiles read from patches.npy at once, about 150 MB


def draw_sample(patches: np.ndarray, labels: np.ndarray, tissue: np.ndarray, animals: np.ndarray,
                per_tile: int, out: Path) -> int:
    """Write a fixed random set of tissue patches per tile: their descriptions, labels, animals and weights."""
    rng = np.random.default_rng(SEED)
    picks, weights = [], []
    for i in range(len(labels)):
        candidates = np.flatnonzero(tissue[i].ravel() >= TISSUE)
        chosen = rng.choice(candidates, min(per_tile, len(candidates)), replace=False)
        picks.append(np.stack([np.full(len(chosen), i), np.sort(chosen)], axis=1))
        # each sampled patch stands for its tile's tissue patches, so a sampled area is an area of patches
        weights.append(np.full(len(chosen), len(candidates) / max(1, len(chosen)), np.float32))
    index = np.concatenate(picks)

    flat = patches.reshape(len(patches), -1, patches.shape[-1])
    x = np.lib.format.open_memmap(out / "sample_x.npy", mode="w+", dtype=patches.dtype,
                                  shape=(len(index), patches.shape[-1]))
    for start in range(0, len(labels), CHUNK):
        rows = (index[:, 0] >= start) & (index[:, 0] < start + CHUNK)
        block = np.asarray(flat[start:start + CHUNK])
        x[rows] = block[index[rows, 0] - start, index[rows, 1]]
    x.flush()
    np.savez(out / "sample.npz", tile=index[:, 0], patch=index[:, 1],
             label=labels.reshape(len(labels), -1)[index[:, 0], index[:, 1]], animal=animals[index[:, 0]],
             weight=np.concatenate(weights))
    return len(index)


def area_error(scores: np.ndarray, truth: np.ndarray, cutoff: float, weight: np.ndarray) -> float:
    """train_necrosis.area_error on a weighted sample: how far the called area lies from the drawn one."""
    return abs(float(np.average(scores >= cutoff, weights=weight) - np.average(truth, weights=weight)))


def fit(job: tuple) -> dict:
    """One head on the sample rows of every animal but the held ones; area errors on each held animal."""
    sample_dir, held, threads = job
    sample = np.load(sample_dir / "sample.npz")
    x = np.load(sample_dir / "sample_x.npy", mmap_mode="r")
    animal, label, weight = sample["animal"], sample["label"], sample["weight"]
    train = ~np.isin(animal, held) & (label >= 0)
    with threadpool_limits(threads):
        features = np.asarray(x[train], dtype=np.float32)
        scaler = StandardScaler().fit(features)
        head = LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)
        head.fit(scaler.transform(features, copy=False), label[train] == 1)
        errors = {}
        for validation in held:
            # every sampled tissue patch, the band included, as the tile rule counts every tile
            check = animal == validation
            scores = head.predict_proba(scaler.transform(np.asarray(x[check], dtype=np.float32)))[:, 1]
            errors[validation] = [area_error(scores, label[check] == 1, c, weight[check]) for c in CUTOFFS]
    return {"held": list(held), "errors": errors, "mean": scaler.mean_, "scale": scaler.scale_,
            "coef": head.coef_[0], "intercept": float(head.intercept_[0]), "n_iter": int(head.n_iter_[0]),
            "n_train": int(train.sum())}


def overlap(truth: np.ndarray, called: np.ndarray) -> tuple[float | None, float | None]:
    """DSC and IoU of two boolean arrays, None where neither holds a positive."""
    both, drawn, kept = int((truth & called).sum()), int(truth.sum()), int(called.sum())
    if not drawn + kept:
        return None, None
    return round(2 * both / (drawn + kept), 4), round(both / (drawn + kept - both), 4)


def code_version() -> str:
    """The git commit the script ran from, marked dirty if the tree had changes and untracked if it is not in it."""
    here = Path(__file__).parent
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=here)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", Path(__file__).name, "train_necrosis.py"],
                             capture_output=True, check=False, cwd=here).returncode == 0
    version = found.stdout.strip()
    return "unknown" if not version else version if tracked else f"{version}-untracked"


def main() -> None:
    """Sample, fit every head and cutoff, score every patch held out, and write heads, scores and metrics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embeddings", type=Path, required=True, help="embed_tiles.py --patches output")
    parser.add_argument("--labels", type=Path, required=True, help="patch_labels.py output")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    parser.add_argument("--per-tile", type=int, default=PER_TILE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty, which a running or finished run leaves; give a new --out")

    meta = json.loads((args.embeddings / "embeddings_meta.json").read_text())
    label_meta = json.loads((args.labels / "patch_labels_meta.json").read_text())
    if not meta.get("patches") or meta["patches"]["grid"] != label_meta["grid"]:
        raise SystemExit(f"{args.embeddings} holds no patch descriptions on the labels' grid")
    if (args.embeddings / "tiles.txt").read_text() != (args.labels / "patch_labels_tiles.txt").read_text():
        raise SystemExit("the labels and the descriptions list different tiles")
    patches = np.load(args.embeddings / "patches.npy", mmap_mode="r")
    labels = np.load(args.labels / "patch_labels.npy")
    tissue = np.load(args.labels / "patch_tissue.npy")
    animals = np.array((args.embeddings / "animals.txt").read_text().split())
    if not len(patches) == len(labels) == len(animals) == meta["n_tiles"]:
        raise SystemExit("descriptions, labels and animals differ in length")
    names = list(meta["animals"])

    args.out.mkdir(parents=True, exist_ok=True)
    n_sample = draw_sample(patches, labels, tissue, animals, args.per_tile, args.out)
    print(f"=== {n_sample:,} patches sampled, {args.per_tile} per tile ===", flush=True)

    # the head without animals a and b checks the cutoff for a held out (on b) and for b held out (on a)
    jobs = [(args.out, pair, args.threads) for pair in combinations(names, 2)]
    jobs += [(args.out, (a,), args.threads) for a in names]
    inner: dict[str, dict[str, list[float]]] = {a: {} for a in names}
    final: dict[str, dict] = {}
    # fresh workers rather than forked ones, so no BLAS thread state is copied into them
    with ProcessPoolExecutor(max(1, args.workers), mp_context=get_context("spawn")) as pool:
        for done, result in enumerate(pool.map(fit, jobs), 1):
            if len(result["held"]) == 2:
                a, b = result["held"]
                inner[a][b], inner[b][a] = result["errors"][b], result["errors"][a]
            else:
                final[result["held"][0]] = result
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} heads fitted", flush=True)

    scores = np.lib.format.open_memmap(args.out / "patch_scores.npy", mode="w+", dtype=np.float16,
                                       shape=labels.shape)
    folds = []
    for a in names:
        mean = np.mean([inner[a][b] for b in names if b != a], axis=0)  # over animals first
        cutoff = float(CUTOFFS[int(np.argmin(mean))])
        head = final[a]
        tiles = np.flatnonzero(animals == a)
        own_scores = np.zeros((len(tiles), *labels.shape[1:]), np.float32)
        for start in range(0, len(tiles), CHUNK):
            part = tiles[start:start + CHUNK]
            x = np.asarray(patches[part], dtype=np.float32)
            logit = ((x - head["mean"]) / head["scale"]) @ head["coef"] + head["intercept"]
            own_scores[start:start + CHUNK] = 1 / (1 + np.exp(-logit))
        scores[tiles] = own_scores  # stored in half precision; the metrics below use the full scores

        own_labels = labels[tiles]
        counted = tissue[tiles] >= TISSUE
        called, truth = own_scores >= cutoff, own_labels == 1
        labelled = own_labels >= 0
        dsc, iou = overlap(truth[labelled], called[labelled])
        dsc_all, iou_all = overlap(truth[counted], called[counted])
        single = len(np.unique(truth[labelled])) < 2
        folds.append({
            "held_out": a,
            "cutoff": cutoff,
            "cutoff_trace": [{"cutoff": float(c), "mean_absolute_area_error": round(float(m), 4)}
                             for c, m in zip(CUTOFFS, mean, strict=True)],
            "n_train_patches": head["n_train"],
            "n_iter": head["n_iter"],
            "hit_max_iter": head["n_iter"] >= MAX_ITER,
            "n_tissue_patches": int(counted.sum()),
            "n_labelled_patches": int(labelled.sum()),
            # the readout: necrotic share of tissue patches, the band counted as the polygons leave it
            "area_true": round(float(truth[counted].mean()), 4),
            "area_predicted": round(float(called[counted].mean()), 4),
            # for information, on the labelled patches only (the band left out)
            "dsc": dsc,
            "iou": iou,
            # over every tissue patch, the band as not necrotic: the area's population, as the tile head counts tiles
            "dsc_all": dsc_all,
            "iou_all": iou_all,
            "balanced_accuracy": round(float(balanced_accuracy_score(truth[labelled], called[labelled])), 4),
            "auroc": None if single else round(float(roc_auc_score(truth[labelled], own_scores[labelled])), 4),
        })
        f = folds[-1]
        print(f"{a}: cutoff {cutoff:.3f}, area {f['area_true']:.1%} drawn, {f['area_predicted']:.1%} called, "
              f"DSC {f['dsc']} ({f['dsc_all']} with the band), IoU {f['iou']} ({f['iou_all']}), "
              f"{f['n_iter']} iterations", flush=True)
    scores.flush()

    np.savez(args.out / "heads.npz", animals=np.array(names), cutoff=[f["cutoff"] for f in folds],
             mean=np.stack([final[a]["mean"] for a in names]), scale=np.stack([final[a]["scale"] for a in names]),
             coef=np.stack([final[a]["coef"] for a in names]), intercept=[final[a]["intercept"] for a in names],
             encoder=meta["model"], grid=meta["patches"]["grid"], code=code_version(),
             note=("score = sigmoid(((x - mean) / scale) @ coef + intercept) on a float16 patch description cast "
                   "to float32; the cutoff was chosen on one pass over whole training-grid tiles, on patches "
                   f"with at least {TISSUE} tissue"))

    necrotic = [f for f in folds if f["area_true"] > 0]
    results = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "head": "LogisticRegression on standardised patch descriptions, as train_necrosis.py on tiles",
        "class_weight": CLASS_WEIGHT,
        "max_iter": MAX_ITER,
        "split": "leave one animal out; the cutoff chosen on the 19 training animals, each left out in turn",
        "cutoff_objective": ("mean absolute area error over the inner animals, on their sampled tissue patches, "
                             "each weighted by its tile's tissue patches over those sampled"),
        "cutoff_grid": [float(c) for c in CUTOFFS],
        "sample": {"per_tile": args.per_tile, "seed": SEED, "n_patches": n_sample,
                   "rule": f"random patches with at least {TISSUE} tissue; trained on those labelled 0 or 1"},
        "area": "necrotic share of the tissue patches of the held-out slide (at least half tissue, lumina out)",
        "metric_caveat": ("DSC and IoU against labels from polygons that stop short of the necrosis edge; dsc and "
                          f"iou leave the {label_meta['band_um']} um band out, dsc_all and iou_all count it as not "
                          "necrotic; for information only"),
        "mean_absolute_area_error": round(float(np.mean([abs(f["area_predicted"] - f["area_true"])
                                                         for f in necrotic])), 4),
        "mean_signed_area_error": round(float(np.mean([f["area_predicted"] - f["area_true"] for f in necrotic])), 4),
        "mean_dsc_necrotic": round(float(np.mean([f["dsc"] for f in necrotic])), 4),
        "mean_iou_necrotic": round(float(np.mean([f["iou"] for f in necrotic])), 4),
        "mean_dsc_all_necrotic": round(float(np.mean([f["dsc_all"] for f in necrotic])), 4),
        "mean_iou_all_necrotic": round(float(np.mean([f["iou_all"] for f in necrotic])), 4),
        "n_necrotic_slides": len(necrotic),
        "folds": folds,
        "files": {
            "heads.npz": "per animal (animals order): scaler mean and scale, coef, intercept, cutoff; its note "
                         "says what the cutoff applies to",
            "patch_scores.npy": "float16, tiles x row x col: each training tile's patches scored by the head "
                                "that never saw its animal",
            "sample_x.npy, sample.npz": ("the training sample: descriptions, tile, patch (row * 28 + col), label, "
                                         "animal, weight"),
        },
        "inputs": {
            "embeddings": str(args.embeddings),
            "embeddings_written": meta["written"],
            "encoder": meta["model"],
            "labels": str(args.labels),
            "labels_written": label_meta["written"],
            "labels_code": label_meta["code"],
            "labels_rule": label_meta["rule"],
        },
    }
    (args.out / "patch_head_results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"=== {len(necrotic)} necrotic slides: area error {results['mean_absolute_area_error']:.2%}, "
          f"DSC {results['mean_dsc_necrotic']} ({results['mean_dsc_all_necrotic']} with the band), "
          f"IoU {results['mean_iou_necrotic']} ({results['mean_iou_all_necrotic']}); written to {args.out} ===")


if __name__ == "__main__":
    main()
