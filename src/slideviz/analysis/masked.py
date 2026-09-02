"""Hide the scanner background, so a layer shows only tissue and nothing else."""

from __future__ import annotations

import dask.array as da
import numpy as np

from slideviz.analysis.tissue import mask_from_level, pick_level

OPAQUE = 255


def alpha_for(level: da.Array, mask: np.ndarray) -> da.Array:
    """The mask as an alpha channel matching one pyramid level, still lazy."""
    height, width = level.shape[:2]
    return da.map_blocks(
        lambda block, block_info=None: _alpha_block(block, block_info, mask, (height, width)),
        level[..., :1],  # one channel, so the result has the shape alpha needs
        dtype=np.uint8,
    )


def _alpha_block(block, block_info, mask, shape_rc) -> np.ndarray:
    """Alpha for one chunk, taking only the slice of the mask that chunk covers."""
    if block_info is None:  # dask's dry run to infer dtype and shape
        return np.zeros(block.shape, np.uint8)

    (row0, row1), (col0, col1) = block_info[0]["array-location"][:2]
    # map this chunk's own pixel range onto the small mask, so nothing global is built
    rows = (np.arange(row0, row1) * mask.shape[0] // shape_rc[0]).clip(0, mask.shape[0] - 1)
    cols = (np.arange(col0, col1) * mask.shape[1] // shape_rc[1]).clip(0, mask.shape[1] - 1)
    return (mask[np.ix_(rows, cols)] * OPAQUE).astype(np.uint8)[..., None]


def to_rgba(levels: list[da.Array], mask: np.ndarray | None = None) -> list[da.Array]:
    """An RGB pyramid with the background made transparent.

    The mask is derived from the level closest to `tissue.TARGET_EDGE_PX` unless one
    is supplied, so a caller that already has a mask does not recompute it.
    """
    if mask is None:
        index = pick_level(levels)
        mask = mask_from_level(np.asarray(levels[index]))

    return [da.concatenate([level, alpha_for(level, mask)], axis=-1) for level in levels]
