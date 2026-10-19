"""Per-patch necrosis map over a whole slide: overlapping tiles embedded and scored on the fly, never stored.

    uv run python predict_slide_patches.py --heads <train_patch_head.py output> \
        --embeddings <its training cache> --tiles <export_field_tiles.py --slide output> \
        --out <dir> --device cuda:0

The whole-slide form of predict_field_patches.py: every tile passes the frozen encoder once, its patch
descriptions are rounded to half precision as the training ones were and scored by the head fitted without
the slide's animal, and each pixel's map value is the mean of the scores of the tiles covering it. The
descriptions themselves (about 90 GB a slide) are not kept. The head must first reproduce its stored
training scores bit for bit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
from embed_tiles import BATCH, DTYPE, INPUT_PX, MODEL, WORKERS, TileSet, build_transform
from predict_field_patches import check_head, pixel_patch, score
from torch.utils.data import DataLoader


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
    """Check the slide's head, embed and score every tile, average onto the slide, and write the map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heads", type=Path, required=True, help="train_patch_head.py output")
    parser.add_argument("--embeddings", type=Path, required=True, help="the training cache the heads were fitted on")
    parser.add_argument("--tiles", type=Path, required=True, help="export_field_tiles.py --slide output, one slide")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    heads = np.load(args.heads / "heads.npz")
    results = json.loads((args.heads / "patch_head_results.json").read_text())
    names = heads["animals"].tolist()
    label_meta = json.loads((Path(results["inputs"]["labels"]) / "patch_labels_meta.json").read_text())
    trained_on = {(lv, sz) for lv in label_meta["geometry"]["level"] for sz in label_meta["geometry"]["size_px"]}
    train_meta = json.loads((args.embeddings / "embeddings_meta.json").read_text())
    manifest = json.loads((args.tiles / "manifest.json").read_text())
    rows = manifest["tiles"]
    ((animal, geometry),) = manifest["fields"].items()
    if not geometry.get("whole_slide"):
        raise SystemExit(f"{args.tiles} holds a field, not a whole slide")
    if str(heads["encoder"]) != MODEL or train_meta["input_px"] != INPUT_PX or train_meta.get("stain_matched"):
        raise SystemExit("the heads were fitted on descriptions made differently from this script's")
    if {(r["level"], r["size_px"]) for r in rows} - trained_on:
        raise SystemExit(f"tiles are not cut like the training tiles {sorted(trained_on)}")
    out = args.out / f"{animal}_patch_slide.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; give a new --out")

    k = names.index(animal)
    cutoff = float(heads["cutoff"][k])
    train_patches = np.load(args.embeddings / "patches.npy", mmap_mode="r")
    train_animals = np.array((args.embeddings / "animals.txt").read_text().split())
    check = check_head(heads, k, animal, train_patches, train_animals,
                       np.load(args.heads / "patch_scores.npy", mmap_mode="r"))
    print(f"=== {animal}: head reproduces its stored scores bit for bit; {len(rows):,} tiles ===", flush=True)

    encoder = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(args.device, dtype=DTYPE)
    config = timm.data.resolve_model_data_config(encoder)
    prefix, grid, dim = encoder.num_prefix_tokens, tuple(encoder.patch_embed.grid_size), encoder.num_features
    if list(grid) != heads["grid"].tolist():
        raise SystemExit("the encoder's patch grid differs from the heads'")
    loader = DataLoader(TileSet(args.tiles, rows, build_transform(config["mean"], config["std"])),
                        batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True,
                        prefetch_factor=4 if args.workers else None)

    size = rows[0]["size_px"]
    top, left = min(r["y"] for r in rows), min(r["x"] for r in rows)
    shape = (max(r["y"] for r in rows) + size - top, max(r["x"] for r in rows) + size - left)
    total, count = np.zeros(shape, np.float64), np.zeros(shape, np.uint16)
    expand = pixel_patch(size, grid[0])
    start, done = time.time(), 0
    with torch.inference_mode():
        for batch, index in loader:
            tokens = encoder.forward_features(batch.to(args.device, dtype=DTYPE, non_blocking=True))
            # rounded to half precision as embed_tiles.py --patches stores the training descriptions
            patches = tokens[:, prefix:].reshape(len(batch), *grid, dim).to(torch.float16).cpu().numpy()
            for i, patch_scores in zip(index.tolist(), score(patches, np.arange(len(batch)), heads, k), strict=True):
                y, x = rows[i]["y"] - top, rows[i]["x"] - left
                total[y:y + size, x:x + size] += patch_scores[np.ix_(expand, expand)]
                count[y:y + size, x:x + size] += 1
            done += len(batch)
            if done % (args.batch * 40) < args.batch or done == len(rows):
                print(f"  {done:>8,}/{len(rows):,}  {done / (time.time() - start):.0f} tiles/s", flush=True)

    seen = count > 0
    mean = np.where(seen, total / np.maximum(count, 1), np.nan).astype(np.float32)
    meta = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "animal": animal,
        "slide": geometry["slide"],
        "head": f"heads.npz index {k}, fitted without {animal}",
        "cutoff": cutoff,
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
    print(f"=== {animal}: {meta['called_share_of_covered']:.1%} of covered pixels called, "
          f"{meta['seconds']:.0f} s, written to {out} ===")


if __name__ == "__main__":
    main()
