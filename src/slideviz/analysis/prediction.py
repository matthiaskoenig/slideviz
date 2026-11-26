"""The per-pixel necrosis call maps: read, placed on an H&E pyramid level, shown in napari; and the lumen masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slideviz.analysis.tiling import pick_level_um
from slideviz.analysis.tissue import pyramid_factor

# necrosis_slides.py call maps: 1 necrotic, 0 not, this value where nothing was measured (glass, lumen, no tile)
CALL_NOT_MEASURED = 255
NECROSIS_RECORD = "necrosis_slides.json"  # the run's areas, beside its maps
CALL_BAND_ROWS = 256  # level rows binned at a time, which keeps the temporary arrays to tens of MB

# the necrotic call as a see-through cyan area; over the gold CYP2E1 steps it mixes to green
CALL_RGB = (0.0, 234 / 255, 248 / 255)
CALL_OPACITY = 0.42
# a necrosis call map is shown on the H&E level nearest this, about 80 MB for a mouse slide
CALL_SHOWN_UM = 0.9
# or on the first coarser level within this many pixels; a rat slide's 0.9 um level holds about 600 MB
CALL_SHOWN_MAX_PX = 160_000_000
# how napari draws a necrotic call: 1 in see-through cyan, 0 clear, no blend between the two
CALL_LOOK = {
    "colormap": {"colors": [[0.0, 0.0, 0.0, 0.0], [*CALL_RGB, 1.0]], "controls": [0.0, 0.5, 1.0],
                 "interpolation": "zero", "name": "necrotic call"},
    "contrast_limits": (0.0, 1.0),
    "opacity": CALL_OPACITY,
    "blending": "translucent_no_depth",
    "interpolation2d": "nearest",  # a yes/no map, a blend between the two would be a third value
}


def read_lumen(path: Path) -> tuple[np.ndarray, float]:
    """A slide's lumen mask and its um/px from a necrosis_lumen.py file."""
    with np.load(path) as stored:
        return stored["lumen"], float(stored["lumen_um"])


def read_calls(path: Path, key: str = "call_E") -> tuple[np.ndarray, dict]:
    """A necrosis_slides.py call map and its record."""
    with np.load(path) as stored:
        return stored[key], json.loads(str(stored["meta"]))


def bin_calls(
    calls: np.ndarray, origin_level_rc: tuple[int, int], factor: int, start_rc: tuple[int, int], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Necrotic and measured masks of a call map on a grid `factor` times coarser than the calls' level.

    Grid pixel (r, c) holds the calls of level pixels start_rc + (r, c) * factor + [0, factor) on each axis; the call
    map's pixel (0, 0) is level pixel origin_level_rc. A grid pixel is measured when at least half of its calls are,
    and necrotic when measured and at least half of its measured calls are necrotic.
    """
    height, width = shape
    top, left = origin_level_rc
    necrotic, measured = np.zeros(shape, bool), np.zeros(shape, bool)
    # a band of grid rows reads its call rows into a canvas starting at the band's first call
    first_col = start_rc[1] - left
    c0, c1 = max(first_col, 0), min(first_col + width * factor, calls.shape[1])
    for q0 in range(0, height, CALL_BAND_ROWS):
        q1 = min(q0 + CALL_BAND_ROWS, height)
        first_row = start_rc[0] + q0 * factor - top
        r0, r1 = max(first_row, 0), min(first_row + (q1 - q0) * factor, calls.shape[0])
        if r1 <= r0 or c1 <= c0:
            continue
        band = np.full(((q1 - q0) * factor, width * factor), CALL_NOT_MEASURED, np.uint8)
        band[r0 - first_row:r1 - first_row, c0 - first_col:c1 - first_col] = calls[r0:r1, c0:c1]
        blocks = band.reshape(q1 - q0, factor, width, factor)
        counted = (blocks != CALL_NOT_MEASURED).sum(axis=(1, 3))
        called = (blocks == 1).sum(axis=(1, 3))
        measured[q0:q1] = 2 * counted >= factor * factor
        necrotic[q0:q1] = measured[q0:q1] & (2 * called >= counted)
    return necrotic, measured


def calls_on_level(
    path: Path, level_shapes: list[tuple[int, int]], index: int, key: str = "call_E"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Necrotic and measured masks of a call map on H&E level `index`, each pixel by the majority of the calls in it."""
    calls, meta = read_calls(path, key)
    factor = pyramid_factor(level_shapes[meta["level"]][1], level_shapes[index][1])
    # call pixel p falls in level pixel (p + factor // 2) // factor, the exact mapping tissue.on_level uses the other way
    start = -(factor // 2)
    necrotic, measured = bin_calls(calls, meta["origin_level_rc"], factor, (start, start), level_shapes[index])
    return necrotic, measured, meta


# the um/px of an H&E pyramid level is the level-0 size times a width ratio, which rounds: levels 1 and 3 of one slide
# are 4 apart by index but up to 1.5e-4 apart from 4 in um/px; a ratio this close to a whole number is that number
PYRAMID_RTOL = 1e-3


def whole_multiple(value: float, unit: float, what: str) -> int:
    """value / unit as the whole number it is within PYRAMID_RTOL (relative to the ratio), an error otherwise."""
    ratio = value / unit
    if abs(ratio - round(ratio)) > PYRAMID_RTOL * max(abs(ratio), 1.0):
        raise ValueError(f"{what}: {value} is not a whole multiple of the call map's {unit} um")
    return round(ratio)


def calls_on_grid(
    calls: np.ndarray, meta: dict, um: float, origin_rc_um: tuple[float, float], rows: slice, cols: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Necrotic and measured masks of a call map over rows x cols of a grid in the H&E frame, as calls_on_level.

    The grid's pixel (r, c) is centred at origin_rc_um + (r, c) * um, a call map pixel (i, j) at (origin_level_rc +
    (i, j)) * um_per_px of its record; the grid's pixel is a whole multiple of the calls', and its origin lies on a
    call pixel centre, both within PYRAMID_RTOL: the grids are levels of one pyramid. Each grid pixel takes the calls whose centres are nearest to its own, the coarse pixel q
    centred on fine pixel factor * q of tissue.on_level.
    """
    call_um = float(meta["um_per_px"])
    factor = whole_multiple(um, call_um, "the grid's pixel size")
    shift = [whole_multiple(o, call_um, "the grid's origin") for o in origin_rc_um]
    start = (shift[0] + rows.start * factor - factor // 2, shift[1] + cols.start * factor - factor // 2)
    shape = (rows.stop - rows.start, cols.stop - cols.start)
    return bin_calls(calls, meta["origin_level_rc"], factor, start, shape)


def call_level(path: Path, level_shapes: list[tuple[int, int]], level0_um: float) -> tuple[np.ndarray, float]:
    """A necrosis call map's necrotic mask on the H&E level nearest CALL_SHOWN_UM within CALL_SHOWN_MAX_PX, and its um/px."""
    index = pick_level_um(level_shapes, level0_um, CALL_SHOWN_UM)
    while index + 1 < len(level_shapes) and level_shapes[index][0] * level_shapes[index][1] > CALL_SHOWN_MAX_PX:
        index += 1
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
