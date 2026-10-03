"""DAB signal quantification for CYP2E1 slides.

Stain optical density against each slide's own glass, separated with one fixed set
of haematoxylin-DAB vectors (Ruifrok and Johnston 2001) for every slide.

Brownness per pixel, (red - blue) / blue following DAB-quant (Fridovich-Keil et al.
2022), is the earlier readout and still feeds the viewer layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from skimage.color import hed_from_rgb

from slideviz.analysis.tissue import mask_from_level, pick_level
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
