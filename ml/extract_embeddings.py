"""Embed image patches with a frozen encoder, cached to disk.

    uv run python extract_embeddings.py --zips /data/michelle/hepatobench/zips \
        --out /data/michelle/hepatobench/embeddings
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ungated, and PFM-DenseBench found it beats encoders 14x its size
MODEL = "hf-hub:1aurent/vit_small_patch8_224.lunit_dino"

# the order fixes the label integers; a cache written under one order is unreadable under another
CLASSES = ("01_TUM", "02_FIB", "03_INF", "04_NEC", "05_NOR", "06_REA", "07_STE")

# timm's default centre-crops to 90%, discarding 18.4% of a patch whose edges carry tissue
INPUT_PX = 224

# fp16 is 3x faster; the full pass is 164 s either way, so keep the baseline plain
DTYPE = torch.float32

BATCH = 256  # 3.7 GiB of 23.5; throughput is flat above this
WORKERS = 16  # PNG decoding is the bottleneck, not the encoder


class PatchZips(Dataset):
    """Every patch across the class zips, as (tensor, label, filename, usable)."""

    def __init__(self, zip_dir: Path, transform: T.Compose):
        """Index every png member across the class zips."""
        self.zip_dir = zip_dir
        self.transform = transform
        self.index: list[tuple[int, str]] = []
        for label, name in enumerate(CLASSES):
            path = zip_dir / f"{name}.zip"
            if not path.exists():
                raise FileNotFoundError(f"no {path.name} in {zip_dir}")
            with zipfile.ZipFile(path) as archive:
                members = sorted(n for n in archive.namelist() if n.endswith(".png"))
            self.index += [(label, member) for member in members]
        # opened lazily: a ZipFile cannot be pickled, so it must not exist when DataLoader forks
        self._handles: dict[int, zipfile.ZipFile] = {}

    def __len__(self) -> int:
        """Number of indexed patches."""
        return len(self.index)

    def _archive(self, label: int) -> zipfile.ZipFile:
        """Open handle for one class zip, created on first use."""
        if label not in self._handles:
            self._handles[label] = zipfile.ZipFile(self.zip_dir / f"{CLASSES[label]}.zip")
        return self._handles[label]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str, bool]:
        """One transformed patch, flagged unusable if it will not decode."""
        label, member = self.index[i]
        raw = self._archive(label).read(member)
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            # 02_FIB_3355.png is zero bytes in the published archive and killed a whole pass
            return torch.zeros(3, INPUT_PX, INPUT_PX), label, member, False
        return self.transform(image), label, member, True


def build_transform(mean: tuple, std: tuple) -> T.Compose:
    """Resize a patch to the encoder's input size, keeping all of it."""
    return T.Compose([
        T.Resize((INPUT_PX, INPUT_PX), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def main() -> None:
    """Embed every patch once and write the cache with its provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", type=Path, required=True, help="directory of class zips")
    parser.add_argument("--out", type=Path, required=True, help="where to write the cache")
    parser.add_argument("--device", default="cuda:0", help="cuda:N, or cpu")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many patches")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {MODEL} ===")
    encoder = timm.create_model(MODEL, pretrained=True, num_classes=0)  # drops the classifier
    encoder.eval().to(args.device, dtype=DTYPE)
    config = timm.data.resolve_model_data_config(encoder)
    parameters = sum(p.numel() for p in encoder.parameters())
    print(f"  {parameters:,} parameters, input {INPUT_PX}px, {DTYPE}")

    dataset = PatchZips(args.zips, build_transform(config["mean"], config["std"]))
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"  {len(dataset):,} patches across {len(CLASSES)} classes")

    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, prefetch_factor=4 if args.workers else None,
    )

    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    skipped: list[str] = []
    seen = 0
    start = time.time()

    print(f"=== embedding on {args.device} ===")
    with torch.inference_mode():
        for batch, label, member, usable in loader:
            out = encoder(batch.to(args.device, dtype=DTYPE, non_blocking=True))
            keep = usable.numpy()
            skipped += [m for m, ok in zip(member, keep, strict=True) if not ok]
            embeddings.append(out.float().cpu().numpy()[keep])
            labels.append(label.numpy()[keep])
            names += [m for m, ok in zip(member, keep, strict=True) if ok]
            seen += len(label)
            if seen % (args.batch * 20) == 0 or seen >= total:
                rate = seen / (time.time() - start)
                print(f"  {seen:>7,}/{total:,}  {rate:>6.0f} patches/s", flush=True)
            if args.limit and seen >= args.limit:
                break

    matrix = np.concatenate(embeddings)
    label_array = np.concatenate(labels)
    elapsed = time.time() - start
    print(f"  {seen:,} patches in {elapsed:.0f} s ({seen / elapsed:.0f}/s)")
    if skipped:
        print(f"  {len(skipped)} skipped, undecodable: {', '.join(sorted(skipped))}")

    # npy memory-maps, so a head reads the cache without loading all of it
    np.save(args.out / "embeddings.npy", matrix)
    np.save(args.out / "labels.npy", label_array)
    (args.out / "patches.txt").write_text("\n".join(names) + "\n")  # row to source patch

    meta = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL,
        "model_parameters": parameters,
        "embedding_dim": int(matrix.shape[1]),
        "n_patches": int(matrix.shape[0]),
        "classes": list(CLASSES),
        "class_counts": {CLASSES[i]: int(n) for i, n in enumerate(np.bincount(label_array))},
        "input_px": INPUT_PX,
        "transform": f"resize {INPUT_PX}x{INPUT_PX} bicubic, no crop",
        "normalize": {"mean": list(config["mean"]), "std": list(config["std"])},
        "dtype": str(DTYPE),
        "device": args.device,
        "batch": args.batch,
        "source": str(args.zips),
        "skipped_undecodable": sorted(skipped),
        "seconds": round(elapsed, 1),
        "torch": torch.__version__,
        "timm": timm.__version__,
        # filenames carry only a class and an index, so a split by slide is impossible
        "patient_ids_available": False,
    }
    (args.out / "embeddings_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"=== wrote {matrix.shape[0]:,} x {matrix.shape[1]} to {args.out} ===")
    for name in ("embeddings.npy", "labels.npy", "patches.txt", "embeddings_meta.json"):
        print(f"  {name:<22} {(args.out / name).stat().st_size / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()
