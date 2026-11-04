"""Per-patch necrosis map over a whole slide, from overlapping tiles embedded and scored on the fly.

    uv run python predict_slide_patches.py --heads <train_patch_head.py output> \
        --embeddings <its training cache> --tiles <export_field_tiles.py --slide output> \
        --out <dir> --device cuda:0 [--stain-reference <stain_stats.py output>]

With --stain-reference, each tile's colours are matched onto the reference's target before the encoder (Reinhard
in LAB), from the slide's statistics in the reference, or from a fixed sample of its tiles when it is not listed.
Every tile passes the frozen encoder once, its patch descriptions are rounded to half precision as the training
ones were and scored, and each pixel's map value is the mean of the scores of the tiles covering it. A training
slide is scored by the head fitted without its animal; a slide outside training (another animal or species) by
the mean of every head's score, against the mean of their cutoffs. The descriptions (about 90 GB a slide) are
discarded after scoring. Each head used must first reproduce its stored training scores bit for bit. Tiles must
match the training tiles in pixel size and edge in um within TILE_TOLERANCE.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
from embed_tiles import (
    BATCH,
    DTYPE,
    INPUT_PX,
    MODEL,
    PATCH_DTYPE,
    WORKERS,
    TileSet,
    build_transform,
)
from stain_norm import read_reference, tile_stats
from torch.utils.data import DataLoader
from train_necrosis import code_version
from train_patch_head import CHUNK, patch_score  # check_head scores by this chunk

TILE_TOLERANCE = 0.05  # relative; rat level 1 is 0.4546 um/px, the mouse training tiles' 0.4421


def score(patches: np.ndarray, part: np.ndarray, heads: np.lib.npyio.NpzFile, k: int) -> np.ndarray:
    """Necrosis scores per patch for the tiles in part, by head k, in train_patch_head.py's arithmetic."""
    return patch_score(patches[part], heads["mean"][k], heads["scale"][k], heads["coef"][k], heads["intercept"][k])


def mean_score(patches: np.ndarray, heads: np.lib.npyio.NpzFile, chosen: list[int]) -> np.ndarray:
    """Mean necrosis score per patch over the chosen heads, each folded into one weight column and offset."""
    coef, scale, mean = heads["coef"][chosen], heads["scale"][chosen], heads["mean"][chosen]
    weight = (coef / scale).T.astype(np.float32)  # (dim, heads)
    offset = (heads["intercept"][chosen] - (mean / scale * coef).sum(axis=1)).astype(np.float32)
    logit = np.asarray(patches, np.float32) @ weight + offset
    return (1 / (1 + np.exp(-logit))).mean(axis=-1, dtype=np.float32)


def slide_scores(patches: np.ndarray, heads: np.lib.npyio.NpzFile, chosen: list[int]) -> np.ndarray:
    """Scores per patch: by one head in the training arithmetic, or the mean over several."""
    if len(chosen) == 1:
        return score(patches, np.arange(len(patches)), heads, chosen[0])
    return mean_score(patches, heads, chosen)


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


def load_heads(heads_dir: Path) -> tuple[np.lib.npyio.NpzFile, dict, list[str], dict]:
    """The heads, their results record, their animal names and the pixel size and tile edge in um of their labels."""
    heads = np.load(heads_dir / "heads.npz")
    results = json.loads((heads_dir / "patch_head_results.json").read_text())
    geometry = json.loads((Path(results["inputs"]["labels"]) / "patch_labels_meta.json").read_text())["geometry"]
    um_per_px = float(np.median(list(geometry["um_per_px"].values())))
    (size_px,) = geometry["size_px"]
    return heads, results, heads["animals"].tolist(), {"um_per_px": um_per_px, "tile_um": size_px * um_per_px}


def unlike_training(geometry: dict, trained: dict) -> str | None:
    """How the tiles' pixel size or edge differs from the training tiles' beyond TILE_TOLERANCE, or None."""
    edge = geometry["size_px"] * geometry["um_per_px"]
    for name, value, train in (("pixel size", geometry["um_per_px"], trained["um_per_px"]),
                               ("tile edge", edge, trained["tile_um"])):
        if abs(value / train - 1) > TILE_TOLERANCE:
            return f"{name} {value:.4f} um is {value / train - 1:+.1%} off the training tiles' {train:.4f} um"
    return None


def training_animal(embeddings_meta: dict, slide: str) -> str | None:
    """The animal a training slide belongs to, from the tile manifest the embeddings were made from; None outside training."""
    rows = json.loads(Path(embeddings_meta["source_manifest"]).read_text())["tiles"]
    return next((r["animal"] for r in rows if r["slide"] == slide), None)


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
    """Check the slide's head, embed and score every tile, average onto the slide, and write the map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heads", type=Path, required=True, help="train_patch_head.py output")
    parser.add_argument("--embeddings", type=Path, required=True, help="the training cache the heads were fitted on")
    parser.add_argument("--tiles", type=Path, required=True, help="export_field_tiles.py --slide output, one slide")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--stain-reference", type=Path, default=None,
                        help="stain_stats.py output: match each tile onto its target before the encoder")
    args = parser.parse_args()

    heads, results, names, trained = load_heads(args.heads)
    train_meta = json.loads((args.embeddings / "embeddings_meta.json").read_text())
    manifest = json.loads((args.tiles / "manifest.json").read_text())
    rows = manifest["tiles"]
    ((_, geometry),) = manifest["fields"].items()
    slide = geometry["slide"]
    if not geometry.get("whole_slide"):
        raise SystemExit(f"{args.tiles} holds a field, not a whole slide")
    if (str(heads["encoder"]) != MODEL or train_meta["input_px"] != INPUT_PX or train_meta.get("stain_matched")
            or (train_meta.get("patches") or {}).get("dtype") != str(PATCH_DTYPE).removeprefix("torch.")):
        raise SystemExit("the heads were fitted on descriptions made differently from this script's")
    unlike = unlike_training(geometry, trained)
    if unlike:
        raise SystemExit(f"tiles are not cut like the training tiles: {unlike}")
    out = args.out / f"{slide}_patch_slide.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; give a new --out")

    animal = training_animal(train_meta, slide)
    chosen = [names.index(animal)] if animal is not None else list(range(len(names)))
    cutoff = float(np.mean(heads["cutoff"][chosen]))
    train_patches = np.load(args.embeddings / "patches.npy", mmap_mode="r")
    train_animals = np.array((args.embeddings / "animals.txt").read_text().split())
    stored = np.load(args.heads / "patch_scores.npy", mmap_mode="r")
    check = {names[k]: check_head(heads, k, names[k], train_patches, train_animals, stored) for k in chosen}
    head = (f"heads.npz index {chosen[0]}, fitted without {animal}" if animal is not None else
            f"mean score of all {len(chosen)} heads in heads.npz, the slide being outside training; "
            "cutoff the mean of theirs")
    stain = None
    if args.stain_reference:
        target, per_slide = read_reference(args.stain_reference)
        listed = slide in per_slide
        stain = {"reference": str(args.stain_reference), "target": target,
                 "source": per_slide[slide] if listed else tile_stats(args.tiles, rows),
                 "source_from": "the reference" if listed else "a sample of this slide's tiles"}
    print(f"=== {slide}: {head}; {len(chosen)} head(s) reproduce their stored scores bit for bit; "
          f"{len(rows):,} tiles; {'stain matched, statistics from ' + stain['source_from'] if stain else 'raw'} ===",
          flush=True)

    encoder = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(args.device, dtype=DTYPE)
    config = timm.data.resolve_model_data_config(encoder)
    prefix, grid, dim = encoder.num_prefix_tokens, tuple(encoder.patch_embed.grid_size), encoder.num_features
    if list(grid) != heads["grid"].tolist():
        raise SystemExit("the encoder's patch grid differs from the heads'")
    tiles = TileSet(args.tiles, rows, build_transform(config["mean"], config["std"]),
                    target=stain["target"] if stain else None, per_slide={slide: stain["source"]} if stain else None)
    loader = DataLoader(tiles,
                        batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True,
                        prefetch_factor=4 if args.workers else None)

    top, left, shape, size = map_frame(rows)
    total, count = np.zeros(shape, np.float64), np.zeros(shape, np.uint16)
    expand = pixel_patch(size, grid[0])
    start, done = time.time(), 0
    with torch.inference_mode():
        for batch, index in loader:
            tokens = encoder.forward_features(batch.to(args.device, dtype=DTYPE, non_blocking=True))
            # rounded to half precision as embed_tiles.py --patches stores the training descriptions
            patches = tokens[:, prefix:].reshape(len(batch), *grid, dim).to(PATCH_DTYPE).cpu().numpy()
            for i, patch_scores in zip(index.tolist(), slide_scores(patches, heads, chosen), strict=True):
                y, x = rows[i]["y"] - top, rows[i]["x"] - left
                total[y:y + size, x:x + size] += patch_scores[np.ix_(expand, expand)]
                count[y:y + size, x:x + size] += 1
            done += len(batch)
            if done % (args.batch * 40) < args.batch or done == len(rows):
                print(f"  {done:>8,}/{len(rows):,}  {done / (time.time() - start):.0f} tiles/s", flush=True)

    seen = count > 0
    mean = mean_map(total, count)
    meta = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(Path(__file__).name),
        "animal": animal,  # the training animal; None outside training
        "slide": slide,
        "head": head,
        "cutoff": cutoff,
        "stain": stain,  # None: raw tiles
        "check": check,
        "level": rows[0]["level"],
        "um_per_px": geometry["um_per_px"],
        "origin_level_rc": [int(top), int(left)],
        "field": geometry,
        "n_tiles": len(rows),
        "tiles_per_pixel_max": int(count.max()),
        "called_share_of_covered": round(float((mean[seen] >= cutoff).mean()), 4),
        "seconds": round(time.time() - start, 1),
        "map": "per level pixel, the mean patch score of every tile covering it; NaN where none does",
        "inputs": {"heads": str(args.heads), "heads_code": str(heads["code"]), "heads_written": results["written"],
                   "embeddings": str(args.embeddings), "tiles": str(args.tiles),
                   "tiles_written": manifest["written"], "device": args.device, "batch": args.batch,
                   "cudnn_tf32": torch.backends.cudnn.allow_tf32, "torch": torch.__version__,
                   "timm": timm.__version__},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mean=mean, count=count.astype(np.uint8), meta=json.dumps(meta))
    print(f"=== {slide}: {meta['called_share_of_covered']:.1%} of covered pixels called, "
          f"{meta['seconds']:.0f} s, written to {out} ===")


if __name__ == "__main__":
    main()
