"""Embed exported slide tiles with the frozen HepatoBench encoder.

    uv run python embed_tiles.py --tiles /data/michelle/mouse/tiles \
        --out /data/michelle/mouse/embeddings

Use --stain-reference to colour-match tiles before encoding.
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
import torchvision.transforms as T
from PIL import Image
from stain_norm import normalise, read_reference
from torch.utils.data import DataLoader, Dataset

# ungated, and PFM-DenseBench found it beats encoders 14x its size
MODEL = "hf-hub:1aurent/vit_small_patch8_224.lunit_dino"

# timm's default centre-crops to 90%, discarding tissue at the tile edge
INPUT_PX = 224

DTYPE = torch.float32
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
    parser.add_argument("--tiles", type=Path, required=True, help="export_tiles.py output")
    parser.add_argument("--out", type=Path, required=True, help="where to write the cache")
    parser.add_argument("--device", default="cuda:0", help="cuda:N, or cpu")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--stain-reference", type=Path,
                       help="stain reference JSON from stain_stats.py; omit to embed raw tiles")
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
            out = encoder(batch.to(args.device, dtype=DTYPE, non_blocking=True))
            embeddings.append(out.float().cpu().numpy())
            order += index.tolist()
            if len(order) % (args.batch * 10) == 0 or len(order) == len(rows):
                rate = len(order) / (time.time() - start)
                print(f"  {len(order):>7,}/{len(rows):,}  {rate:>6.0f} tiles/s", flush=True)

    matrix = np.concatenate(embeddings)
    elapsed = time.time() - start
    print(f"  {len(order):,} tiles in {elapsed:.0f} s ({len(order) / elapsed:.0f}/s)")

    # the loader keeps manifest order, but the index makes that a check rather than a hope
    if order != list(range(len(rows))):
        raise ValueError("tiles came back out of manifest order")

    np.save(args.out / "embeddings.npy", matrix)
    # a fraction, not a class: the threshold is the training stage's choice
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
        # without this two caches look identical and neither says which one was matched
        "stain_matched": bool(args.stain_reference),
        "stain_reference": str(args.stain_reference) if args.stain_reference else None,
        "stain_reference_written": reference_written,
        "device": args.device,
        "source_manifest": str(args.tiles / "manifest.json"),
        "seconds": round(elapsed, 1),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "split_by_animal_possible": True,
    }
    (args.out / "embeddings_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"=== wrote {matrix.shape[0]:,} x {matrix.shape[1]} to {args.out} ===")


if __name__ == "__main__":
    main()
