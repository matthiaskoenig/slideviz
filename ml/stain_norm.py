"""Reinhard stain matching in LAB for embedding tiles.

This duplicates `normalise()` from `src/slideviz/analysis/stain.py`; keep both in sync.
Statistics are read from the stain reference JSON so training and the viewer share a
target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def read_reference(path: Path) -> tuple[dict, dict]:
    """A stain reference file as (target, per-slide statistics), both as mean/sd dicts."""
    record = json.loads(path.read_text())
    return record["target"], record["slides"]


def normalise(tile: np.ndarray, source: dict, target: dict) -> np.ndarray:
    """One tile moved from its slide's colour distribution onto the shared target."""
    lab = rgb2lab(tile.astype(np.float32) / 255)
    src_mean, src_sd = np.asarray(source["mean"]), np.asarray(source["sd"])
    tgt_mean, tgt_sd = np.asarray(target["mean"]), np.asarray(target["sd"])
    # a flat channel would divide by zero, and has nothing to rescale anyway
    scale = np.where(src_sd > 1e-6, tgt_sd / np.maximum(src_sd, 1e-6), 1.0)
    out = (lab - src_mean) * scale + tgt_mean
    return (np.clip(lab2rgb(out), 0, 1) * 255).astype(np.uint8)
