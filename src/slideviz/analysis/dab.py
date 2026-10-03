"""DAB signal quantification for CYP2E1 slides.

Optical density against each slide's glass with fixed Ruifrok vectors. A pixel is
positive when its DAB exceeds the midpoint of its surroundings' low and high level.
Brownness, (red - blue) / blue, is the earlier readout and still feeds the viewer layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.color import hed_from_rgb

from slideviz.analysis.tissue import mask_from_level, pick_level, working_mask
from slideviz.io.reader import open_slide

log = logging.getLogger(__name__)

# tissue pixels above this brownness are DAB positive
CUTOFF = 0.22

# quantiles describing one slide's brownness, enough to put slides on one scale
QUANTILES = (10, 25, 50, 75, 90, 99)

SEED = 0

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


@dataclass(frozen=True)
class Brownness:
    """One slide's brownness distribution over its tissue."""

    median: float
    quantiles: dict[str, float]
    tissue_px: int

    def scaled(self, target: Brownness) -> float:
        """This slide's factor onto a target median, so one cutoff transfers."""
        return target.median / self.median if self.median else 1.0


def brownness(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel brownness: how much redder than blue a pixel is."""
    channels = rgb.astype(np.float32)
    return (channels[..., 0] - channels[..., 2]) / np.maximum(channels[..., 2], 1.0)


def brownness_levels(levels: list, factor: float = 1.0) -> list:
    """A brownness pyramid, computed chunk by chunk as napari draws it."""
    import dask.array as da

    def block(a: np.ndarray) -> np.ndarray:
        """One RGB chunk as brownness, keeping the trailing axis for map_blocks."""
        if not a.size:
            return np.zeros(a.shape[:2] + (1,), dtype=np.float32)
        return (brownness(a) * factor).astype(np.float32)[..., None]

    # drop_axis would collapse the chunk graph, so the channel axis stays width 1
    return [
        da.map_blocks(block, level, dtype=np.float32, chunks=level.chunks[:2] + ((1,),))
        for level in levels
    ]


def slide_brownness(path: Path, scene: int = 0) -> tuple[Brownness, np.ndarray, np.ndarray]:
    """One slide's brownness distribution, with the working level and its tissue mask."""
    _, levels = open_slide(path, scene)
    index = pick_level(levels)
    level = np.asarray(levels[index])
    mask = mask_from_level(level)
    if not mask.any():
        raise ValueError(f"{path.name} has no tissue")

    values = brownness(level)[mask]
    return (
        Brownness(
            median=round(float(np.median(values)), 4),
            quantiles={
                f"p{q}": round(float(np.percentile(values, q)), 4) for q in QUANTILES
            },
            tissue_px=int(mask.sum()),
        ),
        level,
        mask,
    )


def build_reference(slides: list[Path]) -> dict:
    """Every slide's brownness plus the shared median they are scaled onto."""
    found = {}
    for path in sorted(slides):
        stats, _, _ = slide_brownness(path)
        found[path.name.split(".")[0]] = stats
        log.info("%s: median brownness %.3f", path.name, stats.median)

    if not found:
        raise ValueError("no slides to build a reference from")

    # the median of medians, so no single slide pulls the shared scale
    target = float(np.median([s.median for s in found.values()]))
    return {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "brownness (red - blue) / blue, median-scaled across slides",
        "cutoff": CUTOFF,
        "target_median": round(target, 4),
        "slides": {
            name: {"median": s.median, "tissue_px": s.tissue_px, **s.quantiles}
            for name, s in found.items()
        },
    }


def read_reference(path: Path) -> tuple[float, dict[str, float]]:
    """A written reference as its target median and one median per slide."""
    record = json.loads(path.read_text())
    return record["target_median"], {
        name: entry["median"] for name, entry in record["slides"].items()
    }


def positive_area(
    path: Path, cutoff: float = CUTOFF, target_median: float | None = None
) -> dict:
    """DAB positive area as a fraction of tissue, on the shared brightness scale."""
    stats, level, mask = slide_brownness(path)
    values = brownness(level)[mask]

    # scaling the pixels rather than the cutoff keeps one cutoff across slides
    factor = target_median / stats.median if target_median and stats.median else 1.0
    scaled = values * factor

    return {
        "slide": path.name.split(".")[0],
        "tissue_px": stats.tissue_px,
        "median_brownness": stats.median,
        "scale_factor": round(factor, 4),
        "cutoff": cutoff,
        "positive_fraction": round(float((scaled > cutoff).mean()), 4),
        "positive_fraction_unscaled": round(float((values > cutoff).mean()), 4),
        **stats.quantiles,
    }
