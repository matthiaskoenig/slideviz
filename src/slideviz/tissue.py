"""Tissue masks for whole-slide images.

The background of a scanned slide has two parts: synthetic white outside the
scan grid, and real empty glass inside it. Both are background, but only the
step between them is an artifact of where the scanner started, so feature
detectors treat it as the strongest edge on the slide. Masking to tissue keeps
that edge out of registration, and gives area measurements an honest
denominator.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

from slideviz.reader import open_slide

log = logging.getLogger(__name__)

# Work on a level near this edge length: big enough to resolve the tissue
# border, small enough to hold in memory and threshold in well under a second.
TARGET_EDGE_PX = 600

CLOSING_RADIUS = 3  # bridges the gaps torn sections leave along their edge
SPECKLE_PX = 2000  # dust and pen marks, at the working level
HOLE_PX = 2000  # vessel lumina and tears, filled so the mask is solid
KEEP_FRACTION = 0.05  # a piece this much of the largest is a real fragment


def pick_level(levels: list[da.Array], target_edge_px: int = TARGET_EDGE_PX) -> int:
    """Index of the level whose longest edge is closest to the target."""
    edges = [max(level.shape[0], level.shape[1]) for level in levels]
    return min(range(len(edges)), key=lambda i: abs(edges[i] - target_edge_px))


def mask_from_level(level: np.ndarray) -> np.ndarray:
    """Boolean tissue mask for one RGB zoom level."""
    white = (level == 255).all(axis=-1)  # synthetic, written where the grid was not scanned
    rest = ~white
    if not rest.any():
        raise ValueError("level is entirely synthetic white, nothing was scanned")

    grey = level.mean(axis=-1)
    # glass and tissue are two well-separated modes, so Otsu finds the valley
    threshold = threshold_otsu(grey[rest])
    mask = rest & (grey < threshold)

    mask = closing(mask, disk(CLOSING_RADIUS))
    mask = remove_small_holes(mask, max_size=HOLE_PX)
    mask = remove_small_objects(mask, max_size=SPECKLE_PX)
    return keep_large_components(mask)


def keep_large_components(mask: np.ndarray) -> np.ndarray:
    """Drop components far smaller than the largest, keeping genuinely torn pieces."""
    labelled = label(mask)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background is not a component
    if not sizes.any():
        return mask

    keep = np.flatnonzero(sizes >= sizes.max() * KEEP_FRACTION)
    return np.isin(labelled, keep)


def tissue_mask(
    path: Path, scene: int = 0, target_edge_px: int = TARGET_EDGE_PX
) -> tuple[np.ndarray, int, float]:
    """Tissue mask for one slide as (mask, level, um_per_mask_pixel)."""
    info, levels = open_slide(path, scene)
    index = pick_level(levels, target_edge_px)

    level = np.asarray(levels[index])  # the only read; every finer level stays on disk
    mask = mask_from_level(level)

    # the level's own scale, so areas convert without knowing the pyramid
    um_per_px = info.pixel_size_um * (info.width / mask.shape[1])
    log.info("%s scene %d: level %d, tissue %.1f%%", path.name, scene, index, mask.mean() * 100)
    return mask, index, um_per_px


def tissue_area_mm2(mask: np.ndarray, um_per_px: float) -> float:
    """Tissue area in square millimetres."""
    return float(mask.sum()) * (um_per_px / 1000.0) ** 2
