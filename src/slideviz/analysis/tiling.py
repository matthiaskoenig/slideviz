"""A tile grid over a slide, sized in micrometres and filtered to tissue.

The grid is coordinates, not images: tiles are read lazily when they are needed, so
a slide costs nothing until its pixels are.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import dask.array as da
import numpy as np

from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.io.reader import open_slide

log = logging.getLogger(__name__)

# Tissue at 20x, the magnification the HepatoQuant paper tiles at
TARGET_UM_PER_PX = 0.5

# 75 um is the paper's 150 px at 20x, in physical units so it survives a rescan
TILE_UM = 75.0

# a tile below this much tissue is glass or a torn edge
MIN_TISSUE = 0.5


@dataclass(frozen=True)
class Tile:
    """One tile's position on its level, with the provenance to find it again."""

    slide: str
    scene: int
    level: int
    row: int  # index on the tile grid, not a pixel
    col: int
    y: int  # top-left pixel on `level`
    x: int
    size_px: int
    tissue: float  # fraction of the tile covered by tissue


@dataclass(frozen=True)
class Grid:
    """Every tile of one slide, with what produced them."""

    slide: str
    scene: int
    level: int
    size_px: int
    size_um: float
    um_per_px: float
    tiles: list[Tile]

    def to_dict(self) -> dict:
        """The grid as JSON-serialisable provenance."""
        return {
            **{k: v for k, v in asdict(self).items() if k != "tiles"},
            "n_tiles": len(self.tiles),
            "tiles": [asdict(t) for t in self.tiles],
        }


def pick_level_um(
    levels: list[da.Array], px_um: float, target: float = TARGET_UM_PER_PX
) -> int:
    """Index of the level whose pixel size is closest to `target`, in micrometres."""
    scales = [px_um * (levels[0].shape[1] / level.shape[1]) for level in levels]
    return min(range(len(scales)), key=lambda i: abs(scales[i] - target))


def coverage(
    mask: np.ndarray, shape_rc: tuple[int, int], y: int, x: int, size: int
) -> float:
    """Fraction of one tile's footprint covered by a mask held at a coarser scale."""
    rows = (np.arange(y, y + size) * mask.shape[0] // shape_rc[0]).clip(
        0, mask.shape[0] - 1
    )
    cols = (np.arange(x, x + size) * mask.shape[1] // shape_rc[1]).clip(
        0, mask.shape[1] - 1
    )
    return float(mask[np.ix_(rows, cols)].mean())


def build_grid(
    path: Path,
    scene: int = 0,
    tile_um: float = TILE_UM,
    target_um_per_px: float = TARGET_UM_PER_PX,
    min_tissue: float = MIN_TISSUE,
) -> Grid:
    """Tile one slide, keeping the tiles that hold enough tissue."""
    info, levels = open_slide(path, scene)
    level = pick_level_um(levels, info.pixel_size_um, target_um_per_px)

    height, width = levels[level].shape[:2]
    um_per_px = info.pixel_size_um * (levels[0].shape[1] / width)
    size_px = round(tile_um / um_per_px)
    if size_px < 1:
        raise ValueError(f"{tile_um} um is under one pixel at {um_per_px:.4f} um/px")

    mask = mask_from_level(np.asarray(levels[pick_level(levels)]))

    tiles = []
    for row, y in enumerate(range(0, height - size_px + 1, size_px)):
        for col, x in enumerate(range(0, width - size_px + 1, size_px)):
            tissue = coverage(mask, (height, width), y, x, size_px)
            if tissue >= min_tissue:
                tiles.append(
                    Tile(path.name, scene, level, row, col, y, x, size_px, tissue)
                )

    log.info(
        "%s scene %d: level %d, %d px tiles of %.1f um, %d kept",
        path.name,
        scene,
        level,
        size_px,
        tile_um,
        len(tiles),
    )
    return Grid(path.name, scene, level, size_px, tile_um, um_per_px, tiles)


def read_tile(levels: list[da.Array], tile: Tile) -> np.ndarray:
    """One tile's pixels, the only point at which a slide is decoded."""
    level = levels[tile.level]
    return np.asarray(
        level[tile.y : tile.y + tile.size_px, tile.x : tile.x + tile.size_px]
    )
