"""Utilities for displaying per-tile predictions in napari.

Maps use one pixel per tile and are scaled to slide coordinates in micrometres.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

log = logging.getLogger(__name__)

# lumen-grid pixels per map pixel when lumina are cut out, ~3.5 um, so vessels show as holes
LUMEN_POOL = 2


def tile_map(rows: list[dict], values: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """One value per tile as an array indexed by tile row and column."""
    if len(rows) != len(values):
        raise ValueError(f"{len(rows)} tiles against {len(values)} values")
    height = max(r["row"] for r in rows) + 1
    width = max(r["col"] for r in rows) + 1
    grid = np.full((height, width), fill, dtype=float)
    for row, value in zip(rows, values, strict=True):
        grid[row["row"], row["col"]] = value
    return grid


def read_predictions(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """A prediction file's tile rows, predicted scores and annotated fractions."""
    record = json.loads(path.read_text())
    rows = record["tiles"]
    return (
        rows,
        np.array([t["predicted"] for t in rows], dtype=float),
        np.array([t["necrosis"] for t in rows], dtype=float),
    )


def tile_um_of(path: Path, level_um_per_px: float) -> float:
    """Return the tile width in micrometres from the grid's pixel size."""
    record = json.loads(path.read_text())
    return record["size_px"] * level_um_per_px


def read_lumen(path: Path) -> tuple[np.ndarray, float] | None:
    """A slide's lumen mask and its um/px from a necrosis_lumen.py file, or None when absent."""
    if not path.exists():
        return None
    with np.load(path) as stored:
        return stored["lumen"], float(stored["lumen_um"])


def without_lumina(tiles: np.ndarray, tile_um: float, lumen: np.ndarray, lumen_um: float) -> tuple[np.ndarray, float]:
    """A tile map resampled onto the pooled lumen grid with the lumina cut out, and its um/px."""
    height, width = lumen.shape[0] // LUMEN_POOL, lumen.shape[1] // LUMEN_POOL
    pooled = lumen[:height * LUMEN_POOL, :width * LUMEN_POOL].reshape(height, LUMEN_POOL, width, LUMEN_POOL)
    open_share = 1.0 - pooled.mean(axis=(1, 3))
    um = lumen_um * LUMEN_POOL
    # tile centres sit at (index + 0.5) * tile_um, so linear sampling matches the tile map's own blending
    rows = (np.arange(height) + 0.5) * um / tile_um - 0.5
    cols = (np.arange(width) + 0.5) * um / tile_um - 0.5
    values = map_coordinates(tiles, np.meshgrid(rows, cols, indexing="ij"), order=1, mode="constant", cval=0.0)
    return (values * open_share).astype(np.float32), um


def add_prediction_layers(
    viewer,
    path: Path,
    tile_um: float,
    annotated_name: str = "annotated",
    predicted_name: str = "predicted",
    visible: bool = False,
    lumen: tuple[np.ndarray, float] | None = None,
) -> list:
    """Add predicted and annotated tile maps to an open viewer in micrometres.

    Maps use absolute tile row and column indices and the reference stain's frame;
    with a lumen mask and its um/px, they are drawn finer with the lumina cut out.
    """
    rows, predicted, annotated = read_predictions(path)
    maps = {"annotated": tile_map(rows, annotated, fill=0.0), "predicted": tile_map(rows, predicted, fill=0.0)}
    # tiles are square and spaced by their own size, so one tile is one pixel at this scale
    pixel_um = tile_um
    if lumen is not None:
        for key, grid in maps.items():
            maps[key], pixel_um = without_lumina(grid, tile_um, *lumen)
    scale = (pixel_um, pixel_um)
    # Offset by half a pixel because napari centres pixels on their indices.
    offset = (pixel_um / 2, pixel_um / 2)
    common = {
        "scale": scale,
        "translate": offset,
        "units": "um",
        "opacity": 0.5,
        "blending": "additive",  # so the two maps show through each other
        "contrast_limits": (0.0, 1.0),
        "visible": visible,
        # one tile is one pixel, so nearest draws hard squares at every zoom
        "interpolation2d": "linear",
    }
    layers = [
        viewer.add_image(
            maps["annotated"],
            name=annotated_name,
            colormap="green",
            **common,
        ),
        viewer.add_image(
            maps["predicted"],
            name=predicted_name,
            colormap="magenta",
            **common,
        ),
    ]
    log.info("%d tiles as %.1f um pixels%s", len(rows), pixel_um, ", lumina cut out" if lumen is not None else "")
    return layers
