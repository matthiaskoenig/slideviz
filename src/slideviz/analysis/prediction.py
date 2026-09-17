"""Per-tile predictions painted back onto the slide, as a layer napari can show.

A prediction is one number per tile, so the map is the tile grid rather than the slide:
one pixel per tile, held at tile resolution and stretched to slide coordinates by the
layer's scale. The viewer places every layer in micrometres, so the scale is the tile
size in micrometres and the map lands on the tissue without resampling anything.
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


def add_prediction_layers(viewer, path: Path, tile_um: float, name: str = "necrosis") -> None:
    """Add the predicted and annotated tile maps to an open viewer, in micrometres."""
    rows, predicted, annotated = read_predictions(path)
    # tiles are square and spaced by their own size, so one tile is one pixel at this scale
    scale = (tile_um, tile_um)
    viewer.add_image(
        tile_map(rows, annotated),
        name=f"{name} annotated",
        scale=scale,
        units="um",
        colormap="green",
        opacity=0.5,
        blending="translucent",
    )
    viewer.add_image(
        tile_map(rows, predicted),
        name=f"{name} predicted",
        scale=scale,
        units="um",
        colormap="magenta",
        opacity=0.5,
        blending="translucent",
        contrast_limits=(0.0, 1.0),
    )
    log.info("%d tiles as %.0f um pixels", len(rows), tile_um)
