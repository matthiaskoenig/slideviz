"""Compute shared per-slide stain statistics from healthy tiles."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from slideviz.analysis.stain import LabStats, common_target, lab_stats

log = logging.getLogger(__name__)

SAMPLE = 200  # tiles per slide; LAB means converge well before this
SEED = 0  # fixed, so a rerun reproduces the same sample and the same target


@dataclass(frozen=True)
class SlideSample:
    """Which healthy tiles one slide's statistics were computed from."""

    n_clean: int
    n_sampled: int


def clean_tiles(rows: list[dict], slide: str) -> list[dict]:
    """The healthy tiles of one slide, which carry its staining without its necrosis."""
    return [r for r in rows if r["slide"] == slide and r["necrosis"] == 0.0]


def sample_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """A fixed random subset, or all of them when there are fewer than the limit."""
    if len(rows) <= limit:
        return rows
    picked = np.random.default_rng(seed).choice(len(rows), size=limit, replace=False)
    return [rows[i] for i in sorted(picked)]


def read_tiles(tile_dir: Path, rows: list[dict]) -> list[np.ndarray]:
    """Every named tile as an RGB array."""
    return [np.asarray(Image.open(tile_dir / "tiles" / r["tile"]).convert("RGB")) for r in rows]


def slide_stats(
    tile_dir: Path, rows: list[dict], slide: str, limit: int, seed: int
) -> tuple[LabStats, SlideSample]:
    """One slide's LAB distribution, from a sample of its healthy tiles."""
    clean = clean_tiles(rows, slide)
    if not clean:
        raise ValueError(f"{slide}: no tiles with necrosis == 0, so no healthy reference")
    picked = sample_rows(clean, limit, seed)
    stats = lab_stats(read_tiles(tile_dir, picked))
    return stats, SlideSample(n_clean=len(clean), n_sampled=len(picked))


def build(tile_dir: Path, limit: int = SAMPLE, seed: int = SEED) -> dict:
    """Every slide's statistics plus the shared target they are all matched to."""
    manifest = json.loads((tile_dir / "manifest.json").read_text())
    rows = manifest["tiles"]
    slides = sorted({r["slide"] for r in rows})

    stats: dict[str, LabStats] = {}
    samples: dict[str, SlideSample] = {}
    for slide in slides:
        stats[slide], samples[slide] = slide_stats(tile_dir, rows, slide, limit, seed)
        log.info("%s: %d of %d healthy tiles", slide, samples[slide].n_sampled,
                 samples[slide].n_clean)

    target = common_target(stats)
    return {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "tile_dir": str(tile_dir),
        "manifest_written": manifest.get("written"),
        "method": "Reinhard matching in LAB; per-slide mean and sd from healthy tiles",
        "healthy_tiles_only": True,
        "label": manifest.get("label"),
        "sample_per_slide": limit,
        "seed": seed,
        "n_slides": len(slides),
        # the target every slide is matched to; both training and the viewer read this
        "target": asdict(target),
        "slides": {
            s: {**asdict(stats[s]), **asdict(samples[s])} for s in slides
        },
    }


def read(path: Path) -> tuple[LabStats, dict[str, LabStats]]:
    """A written statistics file as the shared target and one LabStats per slide."""
    record = json.loads(path.read_text())
    target = LabStats(tuple(record["target"]["mean"]), tuple(record["target"]["sd"]))
    per_slide = {
        slide: LabStats(tuple(v["mean"]), tuple(v["sd"]))
        for slide, v in record["slides"].items()
    }
    return target, per_slide


def main() -> None:
    """Compute the set's stain statistics and write them where both stages read them."""
    import argparse

    from slideviz.log import setup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=Path, required=True,
                       help="tile directory holding manifest.json and tiles/")
    parser.add_argument("--out", type=Path, required=True, help="statistics JSON to write")
    parser.add_argument("--sample", type=int, default=SAMPLE,
                       help=f"healthy tiles per slide (default {SAMPLE})")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    args = parser.parse_args()
    setup(args.verbose)

    record = build(args.tiles, args.sample, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")

    target = record["target"]
    print(f"\n=== {record['n_slides']} slides, target of the whole set ===")
    print(f"  mean  L {target['mean'][0]:7.2f}  a {target['mean'][1]:7.2f}  "
          f"b {target['mean'][2]:7.2f}")
    print(f"  sd    L {target['sd'][0]:7.2f}  a {target['sd'][1]:7.2f}  "
          f"b {target['sd'][2]:7.2f}")
    print(f"\n{'slide':<26}{'L mean':>9}{'a mean':>9}{'b mean':>9}"
          f"{'L sd':>8}{'clean':>8}")
    for slide, v in record["slides"].items():
        print(f"{slide:<26}{v['mean'][0]:>9.2f}{v['mean'][1]:>9.2f}{v['mean'][2]:>9.2f}"
              f"{v['sd'][0]:>8.2f}{v['n_clean']:>8,}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
