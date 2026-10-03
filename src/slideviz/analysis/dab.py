"""DAB signal quantification for CYP2E1 slides.

Optical density against each slide's glass with fixed Ruifrok vectors. A pixel is
positive when its DAB exceeds the midpoint of its surroundings' low and high level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.color import hed_from_rgb

from slideviz.analysis.tissue import working_mask

# channel order of stain_od
HAEMATOXYLIN, RESIDUAL, DAB = 0, 1, 2

# optical density is scaled by this log, as in skimage.color.rgb2hed, so cutoffs share its units
LOG_FLOOR = np.log(1e-6)


def glass_colour(level: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median RGB of the real glass: outside the tissue, without the synthetic white."""
    glass = ~mask & ~(level == 255).all(axis=-1)
    if not glass.any():
        raise ValueError("no glass outside the tissue to measure")
    return np.median(level[glass], axis=0)


def stain_od(rgb: np.ndarray, glass: np.ndarray) -> np.ndarray:
    """Haematoxylin, residual and DAB optical density per pixel, against the slide's glass."""
    transmitted = np.clip(rgb.astype(np.float32) / glass.astype(np.float32), 1e-6, 1.0)
    od = np.log(transmitted) / LOG_FLOOR
    return np.clip(od @ hed_from_rgb.astype(np.float32), 0, None)


# Relative readout, sizes in um so they hold at any level
GLASS_OD = 0.02  # total optical density below this is glass-like
SMOOTH_UM = 8.0  # about one hepatocyte, so each cell gets one decision
WINDOW_UM = 500.0  # about one lobule, periportal and pericentral zone both inside
STEP_UM = 50.0  # spacing of the windows
CELL_UM = 14.0  # tissue is averaged into cells this size before window statistics
LOW_Q, HIGH_Q = 10, 90  # a window's negative and positive DAB level
MIDPOINT = 0.5  # where between low and high the decision line sits
CONTRAST_MIN = 1.25  # high over low below this is a window without zonation
WINDOW_TISSUE_MIN = 0.3  # share of a window that must be tissue


def glass_like(od: np.ndarray) -> np.ndarray:
    """Pixels whose total optical density is close to the glass."""
    return od.sum(axis=-1) < GLASS_OD


def smooth(dab: np.ndarray, tissue: np.ndarray, um_per_px: float) -> np.ndarray:
    """DAB averaged over about one cell, weighted to tissue so lumina and glass do not dilute it."""
    sigma = SMOOTH_UM / um_per_px
    weight = gaussian_filter(tissue.astype(np.float32), sigma)
    return gaussian_filter(dab * tissue, sigma) / np.maximum(weight, 1e-3)


@dataclass(frozen=True)
class LocalLevels:
    """Low and high DAB of the window around each point of a regular grid over one level."""

    low: np.ndarray
    high: np.ndarray
    spacing_px: float  # grid step in level pixels
    origin_px: float  # level pixel of the first grid point, on both axes
    flat_fraction: float  # windows with tissue but no zonation, which took a neighbour's levels

    def threshold(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """The decision line at level pixel positions, interpolated between grid points."""
        grid = [(rows - self.origin_px) / self.spacing_px, (cols - self.origin_px) / self.spacing_px]
        low = map_coordinates(self.low, grid, order=1, mode="nearest")
        high = map_coordinates(self.high, grid, order=1, mode="nearest")
        return low + MIDPOINT * (high - low)


def local_levels(dab: np.ndarray, tissue: np.ndarray, um_per_px: float) -> LocalLevels:
    """Each window's low and high DAB; windows without tissue or zonation take the nearest valid one."""
    block = max(1, round(CELL_UM / um_per_px))
    h, w = (dab.shape[0] // block) * block, (dab.shape[1] // block) * block
    d = dab[:h, :w].reshape(h // block, block, w // block, block)
    t = tissue[:h, :w].reshape(h // block, block, w // block, block)
    covered = t.sum(axis=(1, 3))
    cells = np.where(covered > block * block / 2, (d * t).sum(axis=(1, 3)) / np.maximum(covered, 1), np.nan)

    win = max(3, round(WINDOW_UM / (um_per_px * block)))
    step = max(1, round(STEP_UM / (um_per_px * block)))
    windows = sliding_window_view(cells, (win, win))[::step, ::step]
    windows = windows.reshape(*windows.shape[:2], -1)
    with np.errstate(all="ignore"):  # windows of only glass give NaN, replaced below
        low = np.nanpercentile(windows, LOW_Q, axis=-1)
        high = np.nanpercentile(windows, HIGH_Q, axis=-1)
    enough = np.isfinite(windows).mean(axis=-1) > WINDOW_TISSUE_MIN
    valid = enough & (high >= CONTRAST_MIN * low)
    if not valid.any():
        raise ValueError("no window with enough tissue and zonation")

    nearest = tuple(distance_transform_edt(~valid, return_distances=False, return_indices=True))
    return LocalLevels(
        low=low[nearest],
        high=high[nearest],
        spacing_px=float(step * block),
        # a window's centre, in level pixels, from its first cell
        origin_px=win * block / 2,
        flat_fraction=round(float((enough & ~valid).sum() / max(enough.sum(), 1)), 4),
    )


def positive_map(dab: np.ndarray, tissue: np.ndarray, levels: LocalLevels, band_px: int = 1024) -> np.ndarray:
    """0 negative, 1 positive, 2 not tissue; evaluated in row bands so memory stays bounded."""
    out = np.full(dab.shape, 2, np.uint8)
    cols = np.arange(dab.shape[1], dtype=np.float32)
    for r0 in range(0, dab.shape[0], band_px):
        r1 = min(r0 + band_px, dab.shape[0])
        rows, cc = np.meshgrid(np.arange(r0, r1, dtype=np.float32), cols, indexing="ij")
        positive = dab[r0:r1] > levels.threshold(rows, cc)
        out[r0:r1] = np.where(tissue[r0:r1], positive, 2)
    return out


def relative_readout(
    rgb: np.ndarray, glass: np.ndarray, outline: np.ndarray, um_per_px: float
) -> tuple[np.ndarray, np.ndarray, LocalLevels]:
    """Positive map, smoothed DAB and window levels for one RGB level of one slide."""
    od = stain_od(rgb, glass)
    tissue = working_mask(outline, glass_like(od), um_per_px)
    dab = smooth(od[..., DAB], tissue, um_per_px)
    del od
    levels = local_levels(dab, tissue, um_per_px)
    return positive_map(dab, tissue, levels), dab, levels

