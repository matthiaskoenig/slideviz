"""Annotation polygons drawn on an overview, converted to per-tile coverage fractions through the recorded pixel sizes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.draw import polygon2mask

from slideviz.analysis.tiling import Grid, Tile

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Coverage:
    """One tile's annotated fraction, alongside the tissue fraction it came with."""

    row: int
    col: int
    y: int  # top-left pixel on the tile grid's level
    x: int
    size_px: int
    tissue: float
    necrosis: float  # fraction of the tile inside an annotation polygon


@dataclass(frozen=True)
class Labels:
    """Every tile of one slide with its coverage, and what produced them."""

    slide: str
    scene: int
    level: int
    size_px: int
    size_um: float
    um_per_px: float
    label: str
    n_polygons: int
    polygon_area_um2: float  # sum of the drawn polygons' areas, independent of the grid
    tiles: list[Coverage]


def read_overview(sidecar: Path) -> dict:
    """The overview export record written beside the PNG that was annotated."""
    return json.loads(sidecar.read_text())


def polygon_area(points: np.ndarray) -> float:
    """Area of one closed polygon by the shoelace formula, in its own units squared."""
    y, x = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def cvat_polygons(points_flat: list[float]) -> np.ndarray:
    """One CVAT polygon's flat x,y list as (row, col) points, the order skimage wants."""
    pairs = np.asarray(points_flat, dtype=float).reshape(-1, 2)
    return pairs[:, ::-1]  # CVAT writes x,y; rows come first here


def to_level(
    points: np.ndarray, overview_um_per_px: float, level_um_per_px: float
) -> np.ndarray:
    """Overview (row, col) points moved onto a level, by the two recorded pixel sizes."""
    return points * (overview_um_per_px / level_um_per_px)


def rasterise(polygons: list[np.ndarray], shape_rc: tuple[int, int]) -> np.ndarray:
    """One boolean mask holding every polygon, at the scale the points are given in."""
    mask = np.zeros(shape_rc, dtype=bool)
    for points in polygons:
        mask |= polygon2mask(shape_rc, points)
    return mask


def tile_coverage(mask: np.ndarray, tile: Tile) -> float:
    """Fraction of one tile's footprint inside the annotation mask."""
    window = mask[tile.y : tile.y + tile.size_px, tile.x : tile.x + tile.size_px]
    return float(window.mean()) if window.size else 0.0


def label_grid(
    grid: Grid,
    polygons: list[np.ndarray],
    overview: dict,
    label: str = "necrosis",
) -> Labels:
    """Convert one slide's overview polygons into a coverage fraction per tile."""
    overview_um_per_px = overview["um_per_px"]
    on_level = [to_level(p, overview_um_per_px, grid.um_per_px) for p in polygons]

    height = round(overview["height"] * overview_um_per_px / grid.um_per_px)
    width = round(overview["width"] * overview_um_per_px / grid.um_per_px)
    mask = rasterise(on_level, (height, width))

    tiles = [
        Coverage(t.row, t.col, t.y, t.x, t.size_px, t.tissue, tile_coverage(mask, t))
        for t in grid.tiles
    ]

    area_um2 = sum(polygon_area(p) for p in polygons) * overview_um_per_px**2

    log.info(
        "%s scene %d: %d polygons over %d tiles, %.1f%% of tiled area",
        grid.slide,
        grid.scene,
        len(polygons),
        len(tiles),
        100 * float(np.mean([t.necrosis for t in tiles])) if tiles else 0.0,
    )
    return Labels(
        grid.slide,
        grid.scene,
        grid.level,
        grid.size_px,
        grid.size_um,
        grid.um_per_px,
        label,
        len(polygons),
        area_um2,
        tiles,
    )
