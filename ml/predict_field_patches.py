"""Per-patch necrosis maps over overlapping field tiles: each field's held-out patch head, scores averaged per pixel.

    uv run python predict_field_patches.py --heads <train_patch_head.py output> \
        --embeddings <its training cache> --field-tiles <export_field_tiles.py output> \
        --field-cache <embed_tiles.py --patches on those tiles> --out <dir>

Every field tile is scored per patch by the head fitted without its animal. A pixel lies in one patch of
every tile that covers it, up to 25 at a 15 um stride, and its map value is the mean of those scores. The
share of those tiles calling it necrotic is kept beside the mean. Before anything is written, each head
must reproduce, bit for bit, its stored scores on the first training tiles of its animal.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from train_necrosis import code_version
from train_patch_head import CHUNK, patch_score  # check_head scores by this chunk


def score(patches: np.ndarray, part: np.ndarray, heads: np.lib.npyio.NpzFile, k: int) -> np.ndarray:
    """Necrosis scores per patch for the tiles in part, by head k, in train_patch_head.py's arithmetic."""
    return patch_score(patches[part], heads["mean"][k], heads["scale"][k], heads["coef"][k], heads["intercept"][k])


def check_head(heads: np.lib.npyio.NpzFile, k: int, animal: str, train_patches: np.ndarray,
               train_animals: np.ndarray, stored: np.ndarray) -> dict:
    """Head k on its animal's first training chunk, which train_patch_head.py scored in one piece; stops unless bit for bit."""
    part = np.flatnonzero(train_animals == animal)[:CHUNK]
    if not part.size:
        raise SystemExit(f"the training cache holds no tiles of {animal} to check its head on")
    differ = int((score(train_patches, part, heads, k).astype(stored.dtype) != stored[part]).sum())
    if differ:
        raise SystemExit(f"{animal}: the head reproduces its stored scores on {part.size} tiles with {differ} "
                         "patches different, nothing written")
    return {"tiles": int(part.size), "patches_different": differ}


def load_heads(heads_dir: Path) -> tuple[np.lib.npyio.NpzFile, dict, list[str], set[tuple[int, int]]]:
    """The heads, their results record, their animal names and the (level, size_px) pairs their labels were cut at."""
    heads = np.load(heads_dir / "heads.npz")
    results = json.loads((heads_dir / "patch_head_results.json").read_text())
    label_meta = json.loads((Path(results["inputs"]["labels"]) / "patch_labels_meta.json").read_text())
    trained_on = {(level, size) for level in label_meta["geometry"]["level"] for size in label_meta["geometry"]["size_px"]}
    return heads, results, heads["animals"].tolist(), trained_on


def map_frame(rows: list[dict]) -> tuple[int, int, tuple[int, int], int]:
    """Top, left, shape and tile size of the map covering these tile rows."""
    size = rows[0]["size_px"]
    top, left = min(r["y"] for r in rows), min(r["x"] for r in rows)
    shape = (max(r["y"] for r in rows) + size - top, max(r["x"] for r in rows) + size - left)
    return top, left, shape, size


def mean_map(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    """total / count per pixel as float32, NaN where count is 0."""
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def pixel_patch(size_px: int, grid: int) -> np.ndarray:
    """The patch row (or column) each tile pixel's centre falls in, with patch edges at r * size_px / grid."""
    return ((np.arange(size_px) + 0.5) * grid / size_px).astype(int)


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

    heads, results, names, trained_on = load_heads(args.heads)

    field_meta = json.loads((args.field_cache / "embeddings_meta.json").read_text())
    manifest = json.loads((args.field_tiles / "manifest.json").read_text())
    rows = manifest["tiles"]
    if not field_meta.get("patches") or field_meta["patches"]["grid"] != heads["grid"].tolist():
        raise SystemExit(f"{args.field_cache} holds no patch descriptions on the heads' grid")
    train_meta = json.loads((args.embeddings / "embeddings_meta.json").read_text())
    same = ("input_px", "transform", "normalize", "stain_matched", "stain_reference")
    if field_meta["model"] != str(heads["encoder"]) or any(field_meta.get(k) != train_meta.get(k) for k in same):
        raise SystemExit("the field descriptions were made differently from the training ones: encoder, "
                         "input size, transform, normalisation or stain matching")
    if (args.field_cache / "tiles.txt").read_text().split() != [r["tile"] for r in rows]:
        raise SystemExit("the field cache does not follow the field manifest's tile order")
    # tile names carry grid positions, the same in every export of a field; the export's stamp ties the cache to these tiles
    if field_meta.get("source_manifest_written") != manifest["written"]:
        raise SystemExit("the field cache was embedded from another export of the field tiles")
    if {(r["level"], r["size_px"]) for r in rows} - trained_on:
        raise SystemExit(f"field tiles are not cut like the training tiles {sorted(trained_on)}")
    grid = field_meta["patches"]["grid"][0]

    train_patches = np.load(args.embeddings / "patches.npy", mmap_mode="r")
    train_animals = np.array((args.embeddings / "animals.txt").read_text().split())
    stored = np.load(args.heads / "patch_scores.npy", mmap_mode="r")
    field_patches = np.load(args.field_cache / "patches.npy", mmap_mode="r")
    field_animals = np.array([r["animal"] for r in rows])

    checks = {animal: check_head(heads, names.index(animal), animal, train_patches, train_animals, stored)
              for animal in manifest["fields"]}
    print(f"=== {len(checks)} heads reproduce their stored scores bit for bit ===", flush=True)

    inputs = {
        "heads": str(args.heads),
        "heads_code": str(heads["code"]),
        "heads_written": results["written"],
        "embeddings": str(args.embeddings),
        "embeddings_written": train_meta["written"],
        "field_tiles": str(args.field_tiles),
        "field_tiles_written": manifest["written"],
        "field_cache": str(args.field_cache),
        "field_cache_written": field_meta["written"],
    }
    run_stamp = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "code": code_version(Path(__file__).name)}

    args.out.mkdir(parents=True, exist_ok=True)
    fields = {}
    for animal, geometry in manifest["fields"].items():
        k = names.index(animal)
        cutoff = float(heads["cutoff"][k])
        own = np.flatnonzero(field_animals == animal)
        if len(own) != geometry["n_tiles"]:
            raise SystemExit(f"{animal}: {len(own)} tiles in the manifest against the field's {geometry['n_tiles']}")
        top, left, shape, size = map_frame([rows[i] for i in own])
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
        mean = mean_map(total, count)
        vote_share = mean_map(votes, count)

        # the field itself, from the first tile centre to the last; the map reaches half a tile beyond it
        first = np.round(np.array(geometry["first_centre"]) - [top, left]).astype(int)
        span = (geometry["grid_shape"][0] - 1) * geometry["stride_px"]
        inside = np.s_[first[0]:first[0] + span, first[1]:first[1] + span]
        counted = seen[inside]
        record = {
            "animal": animal,
            "slide": geometry["slide"],
            "head": f"heads.npz index {k}, fitted without {animal}",
            "cutoff": cutoff,
            "field": geometry,
            "origin_level_rc": [int(top), int(left)],
            "field_box_in_map_rc": [int(first[0]), int(first[1]), int(span)],
            "level": int(rows[own[0]]["level"]),
            "um_per_px": geometry["um_per_px"],
            "n_tiles": len(own),
            "tiles_per_pixel": {"min_seen": int(count[seen].min()), "max": int(count.max())},
            "called_share_mean": round(float((mean[inside][counted] >= cutoff).mean()), 4),
            "called_share_vote": round(float((2 * votes[inside][counted] > count[inside][counted]).mean()), 4),
        }
        np.savez_compressed(args.out / f"{animal}_patch_field.npz", mean=mean, vote_share=vote_share, count=count,
                            meta=json.dumps({**record, **run_stamp, "inputs": inputs}))
        fields[animal] = record
        print(f"{animal}: {len(own):,} tiles, {shape[0]} x {shape[1]} px, up to {record['tiles_per_pixel']['max']} "
              f"tiles a pixel, called {record['called_share_mean']:.1%} by the mean, "
              f"{record['called_share_vote']:.1%} by the vote", flush=True)

    run = {
        **run_stamp,
        "map": ("per level-1 pixel, the mean of the patch scores of every field tile covering it, each tile scored "
                "by the head fitted without its animal; vote_share is the share of those tiles whose patch score "
                "reaches the head's cutoff"),
        "cutoff_note": ("the cutoff was chosen on single tiles of the training grid; called_share_mean applies it to "
                        "the mean, called_share_vote a strict majority of single-tile calls (vote_share > 0.5); for "
                        "information, over the field's covered pixels, glass and lumina included"),
        "files": {"<animal>_patch_field.npz": ("mean and vote_share (float32, NaN where no tile covers), count "
                                               "(tiles per pixel), meta (this record's field entry, the run's "
                                               "stamp and inputs)")},
        "checks": checks,
        "fields": fields,
        "inputs": inputs,
    }
    (args.out / "patch_fields.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"=== {len(fields)} fields written to {args.out} ===")


if __name__ == "__main__":
    main()
