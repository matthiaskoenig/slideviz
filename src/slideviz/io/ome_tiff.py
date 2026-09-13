"""Lazy reading of OME-TIFF whole-slide images written by raw2ometiff.

Returns a list of dask arrays (one per zoom level) that
napari can consume directly as a multiscale image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import tifffile
import zarr

from slideviz.io.czi import SlideInfo

log = logging.getLogger(__name__)

# the scanner's own overview pictures, stored as extra series beside the slide
THUMBNAILS = ("label image", "macro image")

# Tile edge in pixels, the unit napari fetches and caches
TILE = 1024


def image_series(slide: tifffile.TiffFile) -> list[tifffile.TiffPageSeries]:
    """Series holding slide images, with the scanner's thumbnails dropped."""
    return [s for s in slide.series if (s.name or "").lower() not in THUMBNAILS]


def _pixel_size_um(page: tifffile.TiffPage) -> float:
    """Pixel size in micrometres, warning if the two axes disagree."""

    def resolution(code: int) -> float | None:
        value = page.tags[code].value if code in page.tags else None
        # TIFF stores resolution as a rational, in ResolutionUnit per pixel
        return value[0] / value[1] if value and value[0] and value[1] else None

    x_res, y_res = resolution(282), resolution(283)
    if not x_res:
        raise ValueError(f"{page.parent.filename} has no XResolution")

    # raw2ometiff writes pixels per centimetre, so 10000 um over that is the pixel size
    x = 10000 / x_res
    y = 10000 / y_res if y_res else x
    if abs(x - y) > 1e-6 * max(x, y):  # anisotropic, one number cannot describe both
        log.warning("anisotropic pixels, X=%.4f um Y=%.4f um; using X", x, y)
    return x


def _axes_to_yxc(level: da.Array, axes: str) -> da.Array:
    """One level as (y, x, 3), dropping the axes a brightfield slide has one of."""
    order = list(axes)
    for name in ("T", "Z"):
        if name in order:
            level = level[(slice(None),) * order.index(name) + (0,)]
            order.remove(name)

    if "C" in order:  # napari's rgb=True wants channel last
        level = da.moveaxis(level, order.index("C"), -1)
    return level.rechunk({2: -1})  # the three channels of a tile belong in one chunk


def read_info(path: Path, scene: int = 0) -> SlideInfo:
    """Read one scene's geometry and scale without decoding any image data."""
    with tifffile.TiffFile(path) as slide:
        series = image_series(slide)
        if scene >= len(series):
            raise ValueError(
                f"{path.name} has no scene {scene}, only {list(range(len(series)))}"
            )

        level0 = series[scene].levels[0]
        sizes = dict(zip(level0.axes, level0.shape, strict=True))
        return SlideInfo(
            path=path,
            scene=scene,
            n_scenes=len(series),
            # converted pixels start at the origin; the stage offset stayed in the .czi
            x0=0,
            y0=0,
            width=sizes["X"],
            height=sizes["Y"],
            pixel_size_um=_pixel_size_um(level0.keyframe),
        )


def read_pyramid(path: Path, scene: int = 0) -> tuple[SlideInfo, list[da.Array]]:
    """Return (info, zoom levels) for one scene, with level 0 at full resolution."""
    info = read_info(path, scene)

    with tifffile.TiffFile(path) as slide:
        series = image_series(slide)[scene]
        axes = series.levels[0].axes
        n_levels = len(series.levels)
        # the store decodes tiles on demand and outlives the handle it was made from
        store = zarr.open(series.aszarr(), mode="r")

    # one zarr array per level, so only the tiles napari asks for are decoded
    arrays = [store[str(i)] for i in range(n_levels)] if n_levels > 1 else [store]
    return info, [_axes_to_yxc(da.from_zarr(a), axes) for a in arrays]
