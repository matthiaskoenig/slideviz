"""The per-pixel necrosis call maps: read, placed on an H&E pyramid level, shown in napari; and the lumen masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slideviz.analysis.tissue import pyramid_factor

# necrosis_slides.py call maps: 1 necrotic, 0 not, this value where nothing was measured (glass, lumen, no tile)
CALL_NOT_MEASURED = 255
CALL_BAND_ROWS = 256  # level rows binned at a time, which keeps the temporary arrays to tens of MB

# the necrotic call as a see-through cyan area; over the gold CYP2E1 steps it mixes to green
CALL_RGB = (0.0, 234 / 255, 248 / 255)
CALL_OPACITY = 0.42
# a necrosis call map is shown on the H&E level nearest this, about 80 MB for a mouse slide
CALL_SHOWN_UM = 0.9
# how napari draws a necrotic call: 1 in see-through cyan, 0 clear, no blend between the two
CALL_LOOK = {
    "colormap": {"colors": [[0.0, 0.0, 0.0, 0.0], [*CALL_RGB, 1.0]], "controls": [0.0, 0.5, 1.0],
                 "interpolation": "zero", "name": "necrotic call"},
    "contrast_limits": (0.0, 1.0),
    "opacity": CALL_OPACITY,
    "blending": "translucent_no_depth",
    "interpolation2d": "nearest",  # a yes/no map, a blend between the two would be a third value
}


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


def call_level(path: Path, level_shapes: list[tuple[int, int]], level0_um: float) -> tuple[np.ndarray, float]:
    """A necrosis call map's necrotic mask on the H&E level nearest CALL_SHOWN_UM, and that level's um/px."""
    sizes = [level0_um * level_shapes[0][1] / shape[1] for shape in level_shapes]
    index = min(range(len(sizes)), key=lambda i: abs(sizes[i] - CALL_SHOWN_UM))
    necrotic, _, _ = calls_on_level(path, level_shapes, index)
    return necrotic.astype(np.uint8), level0_um * pyramid_factor(level_shapes[0][1], level_shapes[index][1])


def add_call_layer(viewer, necrotic: np.ndarray, um: float, name: str, visible: bool = False) -> object:
    """Add a necrotic mask on one H&E pyramid level to an open viewer as the see-through necrotic area."""
    return viewer.add_image(
        necrotic,
        name=name,
        scale=(um, um),
        # level pixel q is centred on level-0 pixel factor * q, where napari puts it with no shift
        units="um",
        visible=visible,
        **CALL_LOOK,
    )
