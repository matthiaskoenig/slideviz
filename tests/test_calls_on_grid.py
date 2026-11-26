"""A call map binned onto a grid of the H&E frame: the grid must be a pyramid level of the calls, origin on a centre."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("slideviz")
from slideviz.analysis.prediction import calls_on_grid

CALL_UM = 0.5
FAR_UM = 20000.0  # an origin this far from the frame's origin, as a crop deep in a section has


def calls(origin_um: float) -> tuple[np.ndarray, dict]:
    """A 40 x 40 call map whose pixel (0, 0) is centred at origin_um on both axes, necrotic in its left half."""
    start = round(origin_um / CALL_UM)
    data = np.zeros((40, 40), np.uint8)
    data[:, :20] = 1
    return data, {"um_per_px": CALL_UM, "origin_level_rc": [start, start]}


@pytest.mark.parametrize("origin_um", [0.0, FAR_UM])
def test_aligned_origin(origin_um: float) -> None:
    data, meta = calls(origin_um)
    necrotic, measured = calls_on_grid(data, meta, 4 * CALL_UM, (origin_um, origin_um), slice(1, 9), slice(1, 9))
    assert measured.all()
    assert necrotic[:, :4].all() and not necrotic[:, 5:].any()


@pytest.mark.parametrize("origin_um", [0.0, FAR_UM])
@pytest.mark.parametrize("offset_px", [0.25, 0.01, -0.5])
def test_misaligned_origin_rejected(origin_um: float, offset_px: float) -> None:
    """An origin off the call pixel centres, near the frame's origin or far from it, is an error, not rounded."""
    data, meta = calls(origin_um)
    origin = origin_um + offset_px * CALL_UM
    with pytest.raises(ValueError, match="the grid's origin"):
        calls_on_grid(data, meta, 4 * CALL_UM, (origin, origin), slice(0, 8), slice(0, 8))


def test_grid_not_a_pyramid_level_rejected() -> None:
    data, meta = calls(0.0)
    with pytest.raises(ValueError, match="the grid's pixel size"):
        calls_on_grid(data, meta, 3.5 * CALL_UM, (0.0, 0.0), slice(0, 8), slice(0, 8))
