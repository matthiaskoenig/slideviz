"""Per-patch necrosis maps over overlapping field tiles: each field's held-out patch head, scores averaged per pixel.

    uv run python predict_field_patches.py --heads <train_patch_head.py output> \
        --embeddings <its training cache> --field-tiles <export_field_tiles.py output> \
        --field-cache <embed_tiles.py --patches on those tiles> --out <dir>

Every field tile is scored per patch by the head that never saw its animal. A pixel lies in one patch of
every tile that covers it, up to 25 at a 15 um stride, and its map value is the mean of those scores: the
same spot judged in different surroundings, so averaging takes out the tile seams without blurring across
space. The share of those tiles calling it necrotic is kept beside the mean. Before anything is written,
each head must reproduce, bit for bit, its stored scores on the first training tiles of its animal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CHUNK = 256  # tiles scored at once, as train_patch_head.py scores them, so the check repeats its arithmetic


def score(patches: np.ndarray, part: np.ndarray, heads: np.lib.npyio.NpzFile, k: int) -> np.ndarray:
    """Necrosis scores per patch for the tiles in part, by head k, in train_patch_head.py's arithmetic."""
    x = np.asarray(patches[part], dtype=np.float32)
    logit = ((x - heads["mean"][k]) / heads["scale"][k]) @ heads["coef"][k] + heads["intercept"][k]
    return (1 / (1 + np.exp(-logit))).astype(np.float32)


def pixel_patch(size_px: int, grid: int) -> np.ndarray:
    """The patch row (or column) each tile pixel's centre falls in, with patch edges at r * size_px / grid."""
    return ((np.arange(size_px) + 0.5) * grid / size_px).astype(int)


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
    """Check each head, score every field tile per patch, average onto the field, and write one map per field."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heads", type=Path, required=True, help="train_patch_head.py output")
    parser.add_argument("--embeddings", type=Path, required=True, help="the training cache the heads were fitted on")
    parser.add_argument("--field-tiles", type=Path, required=True, help="export_field_tiles.py output")
    parser.add_argument("--field-cache", type=Path, required=True, help="embed_tiles.py --patches on the field tiles")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")

    heads = np.load(args.heads / "heads.npz")
    results = json.loads((args.heads / "patch_head_results.json").read_text())
    names = heads["animals"].tolist()
    label_meta = json.loads((Path(results["inputs"]["labels"]) / "patch_labels_meta.json").read_text())
    trained_on = {(level, size) for level in label_meta["geometry"]["level"] for size in label_meta["geometry"]["size_px"]}

    field_meta = json.loads((args.field_cache / "embeddings_meta.json").read_text())
    manifest = json.loads((args.field_tiles / "manifest.json").read_text())
    rows = manifest["tiles"]
    if not field_meta.get("patches") or field_meta["patches"]["grid"] != heads["grid"].tolist():
        raise SystemExit(f"{args.field_cache} holds no patch descriptions on the heads' grid")
    train_meta = json.loads((args.embeddings / "embeddings_meta.json").read_text())
    if (field_meta["model"], field_meta["input_px"], field_meta["transform"]) != (
            str(heads["encoder"]), train_meta["input_px"], train_meta["transform"]):
        raise SystemExit("the field descriptions come from another encoder or input size than the heads'")
    if (args.field_cache / "tiles.txt").read_text().split() != [r["tile"] for r in rows]:
        raise SystemExit("the field cache does not follow the field manifest's tile order")
    if {(r["level"], r["size_px"]) for r in rows} - trained_on:
        raise SystemExit(f"field tiles are not cut like the training tiles {sorted(trained_on)}")
    grid = field_meta["patches"]["grid"][0]

    train_patches = np.load(args.embeddings / "patches.npy", mmap_mode="r")
    train_animals = np.array((args.embeddings / "animals.txt").read_text().split())
    stored = np.load(args.heads / "patch_scores.npy", mmap_mode="r")
    field_patches = np.load(args.field_cache / "patches.npy", mmap_mode="r")
    field_animals = np.array([r["animal"] for r in rows])

    # each head on its own animal's first training chunk, which train_patch_head.py scored in one piece
    checks = {}
    for animal in manifest["fields"]:
        k = names.index(animal)
        part = np.flatnonzero(train_animals == animal)[:CHUNK]
        again = score(train_patches, part, heads, k).astype(stored.dtype)
        differ = int((again != stored[part]).sum())
        if differ:
            raise SystemExit(f"{animal}: the head reproduces its stored scores on {part.size} tiles with {differ} "
                             "patches different, nothing written")
        checks[animal] = {"tiles": int(part.size), "patches_different": differ}
    print(f"=== {len(checks)} heads reproduce their stored scores bit for bit ===", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = {}
    for animal, geometry in manifest["fields"].items():
        k = names.index(animal)
        cutoff = float(heads["cutoff"][k])
        own = np.flatnonzero(field_animals == animal)
        size = rows[own[0]]["size_px"]
        top, left = min(rows[i]["y"] for i in own), min(rows[i]["x"] for i in own)
        shape = (max(rows[i]["y"] for i in own) + size - top, max(rows[i]["x"] for i in own) + size - left)
        total, votes = np.zeros(shape, np.float64), np.zeros(shape, np.uint16)
        count = np.zeros(shape, np.uint16)
        expand = pixel_patch(size, grid)

        for start in range(0, len(own), CHUNK):
            part = own[start:start + CHUNK]
            for i, patch_scores in zip(part, score(field_patches, part, heads, k), strict=True):
                window = np.s_[rows[i]["y"] - top:rows[i]["y"] - top + size, rows[i]["x"] - left:rows[i]["x"] - left + size]
                spread = patch_scores[np.ix_(expand, expand)]
                total[window] += spread
                votes[window] += spread >= cutoff
                count[window] += 1

        seen = count > 0
        mean = np.where(seen, total / np.maximum(count, 1), np.nan).astype(np.float16)
        vote_share = np.where(seen, votes / np.maximum(count, 1), np.nan).astype(np.float16)
        record = {
            "animal": animal,
            "slide": geometry["slide"],
            "head": f"heads.npz index {k}, fitted without {animal}",
            "cutoff": cutoff,
            "field": geometry,
            "origin_level_px": [int(top), int(left)],
            "level": int(rows[own[0]]["level"]),
            "um_per_px": geometry["um_per_px"],
            "n_tiles": len(own),
            "tiles_per_pixel": {"min_seen": int(count[seen].min()), "max": int(count.max())},
            "called_share_mean": round(float((mean[seen] >= cutoff).mean()), 4),
            "called_share_vote": round(float((vote_share[seen] >= 0.5).mean()), 4),
        }
        np.savez_compressed(args.out / f"{animal}_patch_field.npz", mean=mean, vote_share=vote_share,
                            count=count.astype(np.uint8), meta=json.dumps(record))
        fields[animal] = record
        print(f"{animal}: {len(own):,} tiles, {shape[0]} x {shape[1]} px, up to {record['tiles_per_pixel']['max']} "
              f"tiles a pixel, called {record['called_share_mean']:.1%} by the mean, "
              f"{record['called_share_vote']:.1%} by the vote", flush=True)

    run = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "map": ("per level-1 pixel, the mean of the patch scores of every field tile covering it, each tile scored "
                "by the head fitted without its animal; vote_share is the share of those tiles whose patch score "
                "reaches the head's cutoff"),
        "cutoff_note": ("the cutoff was chosen on single tiles of the training grid; called_share_mean applies it to "
                        "the mean, called_share_vote takes a majority of single-tile calls; for information, "
                        "over every covered pixel, glass and lumina included"),
        "files": {"<animal>_patch_field.npz": ("mean and vote_share (float16, NaN where no tile covers), count "
                                               "(tiles per pixel), meta (this record's field entry)")},
        "checks": checks,
        "fields": fields,
        "inputs": {
            "heads": str(args.heads),
            "heads_code": str(heads["code"]),
            "heads_written": results["written"],
            "embeddings": str(args.embeddings),
            "field_tiles": str(args.field_tiles),
            "field_tiles_written": manifest["written"],
            "field_cache": str(args.field_cache),
            "field_cache_written": field_meta["written"],
        },
    }
    (args.out / "patch_fields.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"=== {len(fields)} fields written to {args.out} ===")


if __name__ == "__main__":
    main()
