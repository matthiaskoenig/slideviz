"""Resample a moving slide pyramid into the reference frame."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
from scipy.ndimage import map_coordinates

from slideviz.data.registration import sha256
from slideviz.data.schema import Registration

log = logging.getLogger(__name__)

CHUNK_PX = 1024
WHITE = 255


@dataclass(frozen=True)
class Field:
    """A displacement field on a grid over the reference slide, in um."""

    dx_um: np.ndarray
    dy_um: np.ndarray
    grid_um_per_px: float


def load_field(sidecar_dir: Path, registration: Registration) -> Field | None:
    """The field a registration points to, or None when absent, altered or stale."""
    nonrigid = registration.nonrigid
    if nonrigid is None:
        return None
    path = sidecar_dir / nonrigid.path
    if not path.exists():
        log.warning("non-rigid field missing: %s", path)
        return None
    if sha256(path) != nonrigid.sha256:
        log.warning("non-rigid field changed since it was attached: %s", path)
        return None
    with np.load(path) as stored:
        # the field applies after one specific matrix
        if not np.allclose(stored["matrix"], registration.matrix):
            log.warning("non-rigid field was computed on another matrix: %s", path)
            return None
        return Field(stored["dx_um"], stored["dy_um"], float(stored["grid_um_per_px"]))


def _chunk(block_info, source: da.Array, inverse: np.ndarray, field: Field,
           reference_um: float, ref_factor: tuple[float, float],
           src_factor: tuple[float, float]) -> np.ndarray:
    """One output chunk: sample the moving level at inverse(matrix) @ (p + d)."""
    (r0, r1), (c0, c1), _ = block_info[None]["array-location"]
    rows, cols = np.mgrid[r0:r1, c0:c1].astype(float)
    y, x = rows * ref_factor[0], cols * ref_factor[1]  # reference level 0 px

    # the field's grid spacing differs from the reference pixel size
    to_grid = reference_um / field.grid_um_per_px
    grid = [y * to_grid, x * to_grid]
    x = x + map_coordinates(field.dx_um, grid, order=1, mode="constant") / reference_um
    y = y + map_coordinates(field.dy_um, grid, order=1, mode="constant") / reference_um

    # reference level 0 -> moving level 0 -> this moving level
    qx = (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / src_factor[1]
    qy = (inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / src_factor[0]

    # synthetic white, as the scanner writes outside its grid, so masking drops it
    out = np.full((r1 - r0, c1 - c0, 3), WHITE, np.uint8)
    top = max(int(np.floor(qy.min())) - 1, 0)
    left = max(int(np.floor(qx.min())) - 1, 0)
    bottom = min(int(np.ceil(qy.max())) + 2, source.shape[0])
    right = min(int(np.ceil(qx.max())) + 2, source.shape[1])
    if bottom <= top or right <= left:  # the chunk lies outside the moving slide
        return out

    crop = np.asarray(source[top:bottom, left:right])
    for channel in range(3):
        out[..., channel] = map_coordinates(
            crop[..., channel], [qy - top, qx - left], order=1, mode="constant", cval=WHITE)
    return out


def nonrigid_levels(moving: list[da.Array], reference_shapes: list[tuple[int, int]],
                    registration: Registration, field: Field,
                    reference_um: float) -> list[da.Array]:
    """The moving pyramid on the reference grid, one level per reference level."""
    inverse = np.linalg.inv(np.array(registration.matrix, float))
    levels = []
    for index, shape in enumerate(reference_shapes):
        source = moving[min(index, len(moving) - 1)]
        ref_factor = (reference_shapes[0][0] / shape[0], reference_shapes[0][1] / shape[1])
        src_factor = (moving[0].shape[0] / source.shape[0],
                      moving[0].shape[1] / source.shape[1])
        template = da.empty((*shape, 3), dtype=np.uint8, chunks=(CHUNK_PX, CHUNK_PX, 3))
        levels.append(template.map_blocks(
            lambda _, block_info=None, s=source, rf=ref_factor, sf=src_factor: _chunk(
                block_info, s, inverse, field, reference_um, rf, sf),
            dtype=np.uint8,
        ))
    return levels
