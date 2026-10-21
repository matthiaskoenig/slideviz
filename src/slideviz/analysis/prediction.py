"""Utilities for displaying per-tile predictions in napari, and for reading the per-pixel necrosis call maps.

Tile maps use one pixel per tile and are scaled to slide coordinates in micrometres.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from slideviz.analysis.tissue import pyramid_factor

log = logging.getLogger(__name__)

# necrosis_slides.py call maps: 1 necrotic, 0 not, this value where nothing was measured (glass, lumen, no tile)
CALL_NOT_MEASURED = 255
CALL_BAND_ROWS = 256  # level rows binned at a time, which keeps the temporary arrays to tens of MB

# lumen-grid pixels per map pixel when lumina are cut out, ~3.5 um, so vessels show as holes
LUMEN_POOL = 2

# the necrotic call as a see-through cyan area; over the gold CYP2E1 steps it mixes to green
CALL_RGB = (0.0, 234 / 255, 248 / 255)
CALL_OPACITY = 0.42
# the call's tile corners rounded for display; the drawn area stays within 0.6 points of the counted one
CALL_ROUND_UM = 20.0


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


def read_calls(path: Path, key: str = "call_E") -> tuple[np.ndarray, dict]:
    """A necrosis_slides.py call map and its record."""
    with np.load(path) as stored:
        return stored[key], json.loads(str(stored["meta"]))


def calls_on_level(
    path: Path, level_shapes: list[tuple[int, int]], index: int, key: str = "call_E"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Necrotic and measured masks of a call map on H&E level `index`, each pixel by the majority of the calls in it."""
    calls, meta = read_calls(path, key)
    factor = pyramid_factor(level_shapes[meta["level"]][1], level_shapes[index][1])
    height, width = level_shapes[index]
    top, left = meta["origin_level_rc"]
    necrotic, measured = np.zeros((height, width), bool), np.zeros((height, width), bool)
    # call pixel p falls in level pixel (p + factor // 2) // factor, the exact mapping tissue.on_level uses
    # the other way; a band of level rows reads its call rows into a canvas starting at that offset
    first_col = -(factor // 2)
    c0, c1 = max(first_col - left, 0), min(first_col + width * factor - left, calls.shape[1])
    for q0 in range(0, height, CALL_BAND_ROWS):
        q1 = min(q0 + CALL_BAND_ROWS, height)
        first_row = q0 * factor - factor // 2
        r0, r1 = max(first_row - top, 0), min(first_row + (q1 - q0) * factor - top, calls.shape[0])
        if r1 <= r0 or c1 <= c0:
            continue
        band = np.full(((q1 - q0) * factor, width * factor), CALL_NOT_MEASURED, np.uint8)
        rows = slice(top + r0 - first_row, top + r1 - first_row)
        band[rows, left + c0 - first_col:left + c1 - first_col] = calls[r0:r1, c0:c1]
        blocks = band.reshape(q1 - q0, factor, width, factor)
        counted = (blocks != CALL_NOT_MEASURED).sum(axis=(1, 3))
        called = (blocks == 1).sum(axis=(1, 3))
        measured[q0:q1] = 2 * counted >= factor * factor
        necrotic[q0:q1] = measured[q0:q1] & (2 * called >= counted)
    return necrotic, measured, meta


def without_lumina(
    tiles: np.ndarray, tile_um: float, lumen: np.ndarray, lumen_um: float, order: int = 1
) -> tuple[np.ndarray, float]:
    """A tile map resampled onto the pooled lumen grid with the lumina cut out, and its um/px; order 0 keeps tiles whole."""
    height, width = lumen.shape[0] // LUMEN_POOL, lumen.shape[1] // LUMEN_POOL
    pooled = lumen[:height * LUMEN_POOL, :width * LUMEN_POOL].reshape(height, LUMEN_POOL, width, LUMEN_POOL)
    open_share = 1.0 - pooled.mean(axis=(1, 3))
    um = lumen_um * LUMEN_POOL
    # tile centres sit at (index + 0.5) * tile_um, so linear sampling matches the tile map's own blending
    rows = (np.arange(height) + 0.5) * um / tile_um - 0.5
    cols = (np.arange(width) + 0.5) * um / tile_um - 0.5
    values = map_coordinates(tiles, np.meshgrid(rows, cols, indexing="ij"), order=order, mode="constant", cval=0.0)
    return (values * open_share).astype(np.float32), um


def add_prediction_layers(
    viewer,
    path: Path,
    tile_um: float,
    predicted_name: str = "predicted",
    visible: bool = False,
    lumen: tuple[np.ndarray, float] | None = None,
) -> list:
    """Add the predicted necrotic area to an open viewer in micrometres.

    The map uses absolute tile row and column indices and the reference stain's frame;
    with a lumen mask and its um/px, it is drawn finer with the lumina cut out.
    The area is the call at the file's nested cutoff, the area the readout counts.
    """
    from napari.utils.colormaps import Colormap

    rows, predicted, _ = read_predictions(path)
    cutoff = json.loads(path.read_text())["cutoff"]
    grid = tile_map(rows, (predicted >= cutoff).astype(float), fill=0.0)
    # tiles are square and spaced by their own size, so one tile is one pixel at this scale
    pixel_um = tile_um
    if lumen is not None:
        # nearest tile, so a pixel is necrotic when it lies in a called tile and mostly outside the lumina
        grid, pixel_um = without_lumina(grid, tile_um, *lumen, order=0)
    call = (gaussian_filter(grid, CALL_ROUND_UM / pixel_um) >= 0.5).astype(np.uint8)
    layers = [
        viewer.add_image(
            call,
            name=predicted_name,
            scale=(pixel_um, pixel_um),
            translate=(pixel_um / 2, pixel_um / 2),  # napari centres pixels on their indices
            units="um",
            colormap=Colormap([[0.0, 0.0, 0.0, 0.0], [*CALL_RGB, 1.0]], controls=[0.0, 0.5, 1.0],
                              interpolation="zero", name="necrotic call"),
            contrast_limits=(0.0, 1.0),
            opacity=CALL_OPACITY,
            blending="translucent_no_depth",
            interpolation2d="nearest",  # a yes/no map, a blend between the two would be a third value
            visible=visible,
        ),
    ]
    log.debug("%d tiles as %.1f um pixels%s", len(rows), pixel_um, ", lumina cut out" if lumen is not None else "")
    return layers
