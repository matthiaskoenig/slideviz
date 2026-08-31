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

from slideviz.czi import SlideInfo

log = logging.getLogger(__name__)

# the scanner's own overview pictures, stored as extra series beside the slide
THUMBNAILS = ("Macro", "Map")

# Tile edge in pixels, the unit napari fetches and caches
TILE = 1024


def image_series(slide: tifffile.TiffFile) -> list[tifffile.TiffPageSeries]:
    """Series holding slide images, with the scanner's thumbnails dropped."""
    return [s for s in slide.series if s.name not in THUMBNAILS]


def _pixel_size_um(page: tifffile.TiffPage) -> float:
    """Pixel size in micrometres, warning if the two axes disagree."""

    def resolution(code: int) -> float | None:
        value = page.tags[code].value if code in page.tags else None
        # TIFF stores resolution as a rational, in ResolutionUnit per pixel
        return value[0] / value[1] if value and value[0] and value[1] else None

    x_res, y_res = resolution(282), resolution(283)
    if not x_res:
        raise ValueError(f"{page.parent.filename} has no XResolution")

    # NDPI records pixels per centimetre, so 10000 um over that is the pixel size
    x = 10000 / x_res
    y = 10000 / y_res if y_res else x
    # NDPI rounds the two resolutions independently (43990 against 43991 on this
    # scanner), so the tolerance is loose enough not to call that anisotropy
    if abs(x - y) > 1e-3 * max(x, y):
        log.warning("anisotropic pixels, X=%.4f um Y=%.4f um; using X", x, y)
    return x


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
            pixel_size_um=_pixel_size_um(level0.keyframe),
        )


def _chunks(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Square tiles across one level, with the remainder as a smaller edge tile."""

    def along(length: int) -> tuple[int, ...]:
        full, rest = divmod(length, TILE)
        return (TILE,) * full + ((rest,) if rest else ())

    return (along(shape[0]), along(shape[1]), (shape[2],))


def _tile(level: zarr.Array, block_info: dict | None = None) -> np.ndarray:
    """One tile, read from the level dask asks for it from."""
    (y0, y1), (x0, x1), (c0, c1) = block_info[None]["array-location"]
    return np.asarray(level[y0:y1, x0:x1, c0:c1])


def _decoded_once(level: da.Array) -> Callable[[], np.ndarray]:
    """A thunk decoding `level` on first call and returning the same array after."""
    held: list[np.ndarray] = []

    def decode() -> np.ndarray:
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

    # One task per tile, avoiding 716k tasks that would exhaust memory.
    levels = [
        da.map_blocks(_tile, level=store[str(i)], dtype=np.uint8, chunks=_chunks(shape))
        for i, shape in enumerate(shapes)
    ]

    # Derive levels below scanner's coarsest, cached and decoded once to save decode time
    coarsest = _decoded_once(levels[-1])
    step, shape = 1, levels[-1].shape
    while max(shape[:2]) > TILE:
        step *= 2
        shape = (-(-shape[0] // 2), -(-shape[1] // 2), shape[2])
        levels.append(
            da.from_delayed(dask.delayed(_decimated)(coarsest, step), shape, np.uint8)
        )
    return info, levels
