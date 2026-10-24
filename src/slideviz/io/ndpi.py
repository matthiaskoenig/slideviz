"""Lazy reading of Hamamatsu .ndpi whole-slide images.

NDPI is a TIFF variant, so tifffile reads it and imagecodecs decodes its JPEG strips.
Returns a list of dask arrays (one per zoom level) that napari can consume directly
as a multiscale image.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import tifffile
import zarr

from slideviz.io.czi import SlideInfo
from slideviz.io.ome_tiff import pixel_size_um

log = logging.getLogger(__name__)

# the scanner's own overview pictures, stored as extra series beside the slide
THUMBNAILS = ("Macro", "Map")

# Tile edge in pixels, the unit napari fetches and caches
TILE = 1024


def image_series(slide: tifffile.TiffFile) -> list[tifffile.TiffPageSeries]:
    """Series holding slide images, with the scanner's thumbnails dropped."""
    return [s for s in slide.series if s.name not in THUMBNAILS]


def read_info(path: Path, scene: int = 0) -> SlideInfo:
    """Read one scene's geometry and scale without decoding any image data."""
    with tifffile.TiffFile(path) as slide:
        series = image_series(slide)
        if scene >= len(series):
            raise ValueError(f"{path.name} has no scene {scene}, only {list(range(len(series)))}")

        level0 = series[scene].levels[0]
        height, width = level0.shape[:2]
        return SlideInfo(
            path=path,
            scene=scene,
            n_scenes=len(series),
            x0=0,
            y0=0,
            width=width,
            height=height,
            pixel_size_um=pixel_size_um(level0.keyframe, tolerance=1e-3),  # NDPI rounds X and Y apart: 43990 against 43991
        )


def _chunks(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Square tiles across one level, with the remainder as a smaller edge tile."""

    def along(length: int) -> tuple[int, ...]:
        """Chunk sizes along one axis: full tiles, then the remainder."""
        full, rest = divmod(length, TILE)
        return (TILE,) * full + ((rest,) if rest else ())

    return (along(shape[0]), along(shape[1]), (shape[2],))


def _tile(level: zarr.Array, block_info: dict | None = None) -> np.ndarray:
    """The block of `level` at the array location dask passes in block_info."""
    (y0, y1), (x0, x1), (c0, c1) = block_info[None]["array-location"]
    return np.asarray(level[y0:y1, x0:x1, c0:c1])


def _decoded_once(level: da.Array) -> Callable[[], np.ndarray]:
    """A thunk decoding `level` on first call and returning the same array after."""
    held: list[np.ndarray] = []

    def decode() -> np.ndarray:
        """The decoded level, computed on the first call and held after."""
        if not held:
            held.append(level.compute())
        return held[0]

    return decode


def _decimated(decode: Callable[[], np.ndarray], step: int) -> np.ndarray:
    """Every `step`-th pixel of the decoded level."""
    return decode()[::step, ::step]


def read_pyramid(path: Path, scene: int = 0) -> tuple[SlideInfo, list[da.Array]]:
    """Return (info, zoom levels) for one scene, with level 0 at full resolution."""
    info = read_info(path, scene)

    with tifffile.TiffFile(path) as slide:
        series = image_series(slide)[scene]
        shapes = [lv.shape for lv in series.levels]
        # the store decodes strips on demand and outlives the handle it was made from
        store = zarr.open(series.aszarr(), mode="r")

    # one dask task per TILE x TILE block, each spanning many native NDPI JPEG tiles
    arrays = [store[str(i)] for i in range(len(shapes))] if len(shapes) > 1 else [store]
    levels = [
        da.map_blocks(_tile, level=array, dtype=np.uint8, chunks=_chunks(shape))
        for array, shape in zip(arrays, shapes, strict=True)
    ]

    # levels below the scanner's coarsest, decimated from one cached decode of it
    coarsest = _decoded_once(levels[-1])
    step, shape = 1, levels[-1].shape
    while max(shape[:2]) > TILE:
        step *= 2
        shape = (-(-shape[0] // 2), -(-shape[1] // 2), shape[2])
        levels.append(
            da.from_delayed(dask.delayed(_decimated)(coarsest, step), shape, np.uint8)
        )
    return info, levels
