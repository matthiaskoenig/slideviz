"""Embed exported slide tiles with the frozen HepatoBench encoder.

    uv run python embed_tiles.py --tiles /data/michelle/mouse/tiles \
        --out /data/michelle/mouse/embeddings

Use --stain-reference to colour-match tiles before encoding, and --patches to also keep
the encoder's description of every patch in a tile, for a necrosis map finer than the tile.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from stain_norm import normalise, read_reference
from torch.utils.data import DataLoader, Dataset

# ungated, and PFM-DenseBench found it beats encoders 14x its size
MODEL = "hf-hub:1aurent/vit_small_patch8_224.lunit_dino"

# timm's default centre-crops to 90%, discarding tissue at the tile edge
INPUT_PX = 224

DTYPE = torch.float32
# half precision halves the patch file, about 75 GB for the 124k training tiles
PATCH_DTYPE = torch.float16
BATCH = 256
WORKERS = 16


class TileSet(Dataset):
    """Every tile in the manifest, as (tensor, index)."""

    def __init__(
        self,
        tile_dir: Path,
        rows: list[dict],
        transform: T.Compose,
        target: dict | None = None,
        per_slide: dict | None = None,
    ):
        """Hold the manifest rows, their directory, and any stain correction to apply."""
        self.tile_dir = tile_dir
        self.rows = rows
        self.transform = transform
        self.target = target
        self.per_slide = per_slide or {}

    def __len__(self) -> int:
        """Number of tiles in the manifest."""
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """One transformed tile, with its manifest row index."""
        row = self.rows[i]
        image = Image.open(self.tile_dir / "tiles" / row["tile"]).convert("RGB")
        if self.target is not None:
            # in memory, so one tile set on disk serves both the raw and matched runs
            matched = normalise(np.asarray(image), self.per_slide[row["slide"]], self.target)
            image = Image.fromarray(matched)
        return self.transform(image), i


def build_transform(mean: tuple, std: tuple) -> T.Compose:
    """Resize a tile to the encoder's input size, keeping all of it."""
    return T.Compose([
        T.Resize((INPUT_PX, INPUT_PX), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def main() -> None:
    """Embed every exported tile once and write the cache with its provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=Path, required=True, help="export_tiles.py or export_field_tiles.py output")
    parser.add_argument("--out", type=Path, required=True, help="where to write the cache")
    parser.add_argument("--device", default="cuda:0", help="cuda:N, or cpu")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--stain-reference", type=Path,
                       help="stain reference JSON from stain_stats.py; omit to embed raw tiles")
    parser.add_argument("--patches", action="store_true",
                       help="also write patches.npy, the last layer's description of every patch")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.tiles / "manifest.json").read_text())
    rows = manifest["tiles"]

    print(f"=== loading {MODEL} ===")
    encoder = timm.create_model(MODEL, pretrained=True, num_classes=0)  # drops the classifier
    encoder.eval().to(args.device, dtype=DTYPE)
    config = timm.data.resolve_model_data_config(encoder)
    parameters = sum(p.numel() for p in encoder.parameters())
    print(f"  {parameters:,} parameters, input {INPUT_PX}px")
    print(f"  {len(rows):,} tiles across {len(manifest['animals'])} animals")

    target, per_slide, reference_written = (None, {}, None)
    if args.stain_reference:
        target, per_slide = read_reference(args.stain_reference)
        # a slide with no statistics would silently go through uncorrected
        missing = sorted({r["slide"] for r in rows} - set(per_slide))
        if missing:
            raise ValueError(f"stain reference covers no statistics for: {', '.join(missing)}")
        reference_written = json.loads(args.stain_reference.read_text()).get("written")
        print(f"=== stain matching against {args.stain_reference.name} ===")
        print(f"  target L {target['mean'][0]:.2f}  a {target['mean'][1]:.2f}  "
              f"b {target['mean'][2]:.2f}, {len(per_slide)} slides")

    # the encoder's output is its prefix tokens (here one, the summary) followed by one token
    # per patch, row by row from the tile's top left
    patches, final = None, args.out / "patches.npy"
    if args.patches:
        prefix, grid, dim = encoder.num_prefix_tokens, tuple(encoder.patch_embed.grid_size), encoder.num_features
        # written under this run's own name and moved into place at the end, so a failed run
        # leaves an earlier patch file and its meta as they were
        partial = args.out / f"patches.npy.{os.getpid()}.partial"
        patches = np.lib.format.open_memmap(partial, mode="w+", dtype=str(PATCH_DTYPE).removeprefix("torch."),
                                            shape=(len(rows), *grid, dim))
        print(f"  keeping {grid[0]}x{grid[1]} patches of {dim}, {patches.nbytes / 1e9:.1f} GB to {final}")

    dataset = TileSet(
        args.tiles, rows, build_transform(config["mean"], config["std"]), target, per_slide
    )
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, prefetch_factor=4 if args.workers else None,
    )

    embeddings: list[np.ndarray] = []
    order: list[int] = []
    start = time.time()

    print(f"=== embedding on {args.device} ===")
    with torch.inference_mode():
        for batch, index in loader:
            batch = batch.to(args.device, dtype=DTYPE, non_blocking=True)
            if patches is None:
                out = encoder(batch)
            else:
                # one pass for both: encoder(batch) is forward_head of these tokens
                tokens = encoder.forward_features(batch)
                if tokens.shape[1] != prefix + grid[0] * grid[1]:
                    raise ValueError(f"{tokens.shape[1]} tokens, expected {prefix} + {grid[0]}x{grid[1]}")
                out = encoder.forward_head(tokens)
                patches[index.numpy()] = (tokens[:, prefix:].reshape(len(batch), *grid, dim)
                                          .to(PATCH_DTYPE).cpu().numpy())
            embeddings.append(out.float().cpu().numpy())
            order += index.tolist()
            if len(order) % (args.batch * 10) == 0 or len(order) == len(rows):
                rate = len(order) / (time.time() - start)
                print(f"  {len(order):>7,}/{len(rows):,}  {rate:>6.0f} tiles/s", flush=True)

    matrix = np.concatenate(embeddings)
    elapsed = time.time() - start
    print(f"  {len(order):,} tiles in {elapsed:.0f} s ({len(order) / elapsed:.0f}/s)")

    # the loader keeps manifest order; the index checks it
    if order != list(range(len(rows))):
        raise ValueError("tiles came back out of manifest order")

    np.save(args.out / "embeddings.npy", matrix)
    if patches is not None:
        patches.flush()
        os.replace(partial, final)
    elif final.exists():
        final.unlink()  # an earlier run's patches, stale against these embeddings
    # a fraction; the training stage sets the threshold
    np.save(args.out / "necrosis.npy", np.array([r["necrosis"] for r in rows], dtype=np.float32))
    np.save(args.out / "tissue.npy", np.array([r["tissue"] for r in rows], dtype=np.float32))
    # the split is by animal, so the animal has to survive into the cache
    (args.out / "animals.txt").write_text("\n".join(r["animal"] for r in rows) + "\n")
    (args.out / "tiles.txt").write_text("\n".join(r["tile"] for r in rows) + "\n")

    meta = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL,
        "model_parameters": parameters,
        "embedding_dim": int(matrix.shape[1]),
        "n_tiles": int(matrix.shape[0]),
        "animals": manifest["animals"],
        "tiles_per_animal": {
            a: sum(1 for r in rows if r["animal"] == a) for a in manifest["animals"]
        },
        "label": "necrosis",
        "label_is_fraction": True,
        "input_px": INPUT_PX,
        "transform": f"resize {INPUT_PX}x{INPUT_PX} bicubic, no crop",
        "normalize": {"mean": list(config["mean"]), "std": list(config["std"])},
        "stain_matched": bool(args.stain_reference),
        "stain_reference": str(args.stain_reference) if args.stain_reference else None,
        "stain_reference_written": reference_written,
        "patches": {
            "file": "patches.npy",
            "shape": ["tile", "row", "col", "dim"],
            "grid": list(grid),
            "dim": dim,
            "dtype": patches.dtype.name,
            "layer": f"last block, then {type(encoder.norm).__name__}",
            "prefix_tokens_dropped": prefix,
            "summary_pool": encoder.global_pool,  # token: embeddings.npy is the first prefix token
            "order": "row r from the tile's top, column c from its left",
        } if patches is not None else None,
        "tile_um": manifest.get("tile_um"),
        "device": args.device,
        # the outputs move by about 0.1 % between batch sizes (TF32 convolutions), and are
        # byte-identical between runs at the same one
        "batch": args.batch,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "source_manifest": str(args.tiles / "manifest.json"),
        "source_manifest_written": manifest.get("written"),
        "seconds": round(elapsed, 1),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "split_by_animal_possible": True,
    }
    (args.out / "embeddings_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"=== wrote {matrix.shape[0]:,} x {matrix.shape[1]} to {args.out} ===")


if __name__ == "__main__":
    main()
