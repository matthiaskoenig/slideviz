"""Reinhard stain matching in LAB for embedding tiles.

This duplicates `normalise()` from `src/slideviz/analysis/stain.py`; keep both in sync.
Statistics are read from the stain reference JSON so training and the viewer share a
target; a slide the reference does not list gets its statistics from a sample of its tiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

SAMPLE = 200  # tiles per slide, as stain_stats.py sampled each mouse slide
SEED = 0


def read_reference(path: Path) -> tuple[dict, dict]:
    """A stain reference file as (target, per-slide statistics), both as mean/sd dicts."""
    record = json.loads(path.read_text())
    return record["target"], record["slides"]


def tile_stats(tile_dir: Path, rows: list[dict], sample: int = SAMPLE, seed: int = SEED) -> dict:
    """Mean and sd per LAB channel over a fixed random sample of one slide's tiles, as lab_stats in stain.py."""
    if len(rows) > sample:
        rows = [rows[i] for i in sorted(np.random.default_rng(seed).choice(len(rows), size=sample, replace=False))]
    lab = np.concatenate([rgb2lab(np.asarray(Image.open(tile_dir / "tiles" / r["tile"]).convert("RGB"),
                                             np.float32) / 255).reshape(-1, 3) for r in rows])
    return {"mean": lab.mean(axis=0).tolist(), "sd": lab.std(axis=0).tolist(), "n_sampled": len(rows),
            "seed": seed}


def normalise(tile: np.ndarray, source: dict, target: dict) -> np.ndarray:
    """One tile moved from its slide's colour distribution onto the shared target."""
    lab = rgb2lab(tile.astype(np.float32) / 255)
    src_mean, src_sd = np.asarray(source["mean"]), np.asarray(source["sd"])
    tgt_mean, tgt_sd = np.asarray(target["mean"]), np.asarray(target["sd"])
    # a flat channel would divide by zero, and has nothing to rescale anyway
    scale = np.where(src_sd > 1e-6, tgt_sd / np.maximum(src_sd, 1e-6), 1.0)
    out = (lab - src_mean) * scale + tgt_mean
    matched = (np.clip(lab2rgb(out), 0, 1) * 255).astype(np.uint8)
    # pure white is the scanner's fill outside the scan grid: no stain, and the background mask finds it by that value
    return np.where((tile == 255).all(axis=-1, keepdims=True), tile, matched)
