"""Utilities for displaying per-tile predictions in napari.

Maps use one pixel per tile and are scaled to slide coordinates in micrometres.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


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


def add_prediction_layers(
    viewer,
    path: Path,
    tile_um: float,
    annotated_name: str = "annotated",
    predicted_name: str = "predicted",
    visible: bool = False,
) -> list:
    """Add predicted and annotated tile maps to an open viewer in micrometres.

    Maps use absolute tile row and column indices and the reference stain's frame.
    """
    rows, predicted, annotated = read_predictions(path)
    # tiles are square and spaced by their own size, so one tile is one pixel at this scale
    scale = (tile_um, tile_um)
    # Offset by half a tile because napari centres pixels on their indices.
    offset = (tile_um / 2, tile_um / 2)
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
            tile_map(rows, annotated, fill=0.0),
            name=annotated_name,
            colormap="green",
            **common,
        ),
        viewer.add_image(
            tile_map(rows, predicted, fill=0.0),
            name=predicted_name,
            colormap="magenta",
            **common,
        ),
    ]
    log.info("%d tiles as %.0f um pixels", len(rows), tile_um)
    return layers
