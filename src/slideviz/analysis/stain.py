"""Per-slide stain normalisation, so a slide's staining cannot stand in for its dose.

Staining varies between batches, and if it happens to correlate with dose the encoder
can read colour instead of morphology. Reinhard matching in LAB moves each slide's
colour distribution onto a shared target: shift by the mean, scale by the standard
deviation, one channel at a time.

The statistics come from HEALTHY tiles only. Necrotic tissue is genuinely paler, so a
slide with more necrosis would otherwise pull its own reference and the correction
would partly erase the signal it is meant to preserve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from skimage.color import lab2rgb, rgb2lab

log = logging.getLogger(__name__)


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    """Three numpy scalars as plain floats, which JSON can write and numpy cannot."""
    a, b, c = (float(v) for v in values)
    return a, b, c


@dataclass(frozen=True)
class LabStats:
    """One slide's colour distribution in LAB, as per-channel mean and spread."""

    mean: tuple[float, float, float]
    sd: tuple[float, float, float]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The two triples as arrays, ready to broadcast over an image."""
        return np.asarray(self.mean), np.asarray(self.sd)


def lab_stats(tiles: list[np.ndarray]) -> LabStats:
    """Mean and standard deviation per LAB channel over a sample of one slide's tiles."""
    lab = np.concatenate([rgb2lab(t.astype(np.float32) / 255).reshape(-1, 3) for t in tiles])
    # plain floats, so the statistics survive json.dumps into a manifest
    return LabStats(_triple(lab.mean(axis=0)), _triple(lab.std(axis=0)))


def common_target(stats: dict[str, LabStats]) -> LabStats:
    """The target every slide is matched to: the average of their distributions."""
    means = np.array([s.mean for s in stats.values()])
    sds = np.array([s.sd for s in stats.values()])
    return LabStats(_triple(means.mean(axis=0)), _triple(sds.mean(axis=0)))


def normalise(tile: np.ndarray, source: LabStats, target: LabStats) -> np.ndarray:
    """One tile moved from its slide's colour distribution onto the shared target."""
    lab = rgb2lab(tile.astype(np.float32) / 255)
    src_mean, src_sd = source.as_arrays()
    tgt_mean, tgt_sd = target.as_arrays()
    # a flat channel would divide by zero, and has nothing to rescale anyway
    scale = np.where(src_sd > 1e-6, tgt_sd / np.maximum(src_sd, 1e-6), 1.0)
    out = (lab - src_mean) * scale + tgt_mean
    return (np.clip(lab2rgb(out), 0, 1) * 255).astype(np.uint8)
