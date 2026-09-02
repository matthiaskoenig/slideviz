"""Lazy reading of OME-Zarr whole-slide images written by bioformats2raw.

Returns a list of dask arrays (one per zoom level) that
napari can consume directly as a multiscale image.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import dask.array as da

from slideviz.io.czi import SlideInfo

log = logging.getLogger(__name__)

# bioformats2raw stores the slide's own thumbnails as extra series (not scenes)
THUMBNAILS = ("label image", "macro image")


def _multiscales(path: Path, series: str) -> dict:
    """The multiscales metadata of one series."""
    return json.loads((path / series / ".zattrs").read_text())["multiscales"][0]


def image_series(path: Path) -> list[str]:
    """Series holding slide images, in order, with the scanner's thumbnails dropped."""
    listed = json.loads((path / "OME" / ".zattrs").read_text())["series"]
    return [s for s in listed if _multiscales(path, s).get("name") not in THUMBNAILS]


def _pixel_size_um(ms: dict) -> float:
    """Pixel size in micrometres from the level-0 scale, warning if the axes disagree."""
    axes = [a["name"] for a in ms["axes"]]
    scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    sizes = dict(zip(axes, scale))

    x, y = sizes["x"], sizes.get("y", sizes["x"])
    if abs(x - y) > 1e-6 * max(x, y):  # anisotropic, one number cannot describe both
        log.warning("anisotropic pixels, X=%.4f um Y=%.4f um; using X", x, y)
    return x  # OME-Zarr stores it in the axis unit, micrometres here


def read_info(path: Path, scene: int = 0) -> SlideInfo:
    """Read one scene's geometry and scale without decoding any image data."""
    series = image_series(path)
    if scene >= len(series):
        raise ValueError(f"{path.name} has no scene {scene}, only {list(range(len(series)))}")

    ms = _multiscales(path, series[scene])
    axes = [a["name"] for a in ms["axes"]]
    shape = da.from_zarr(path / series[scene] / ms["datasets"][0]["path"]).shape
    sizes = dict(zip(axes, shape))

    return SlideInfo(
        path=path,
        scene=scene,
        n_scenes=len(series),
        # converted pixels start at the origin; the stage offset stayed in the .czi
        x0=0,
        y0=0,
        width=sizes["x"],
        height=sizes["y"],
        pixel_size_um=_pixel_size_um(ms),
    )


def read_level(path: Path, series: str, dataset: str, axes: list[str]) -> da.Array:
    """One zoom level as (y, x, 3), lazily, from its chunks on disk."""
    level = da.from_zarr(path / series / dataset)  # chunked as written, nothing read yet

    # drop the axes a brightfield slide has only one of, keeping c for the RGB move
    for name in ("t", "z"):
        if name in axes:
            level = level[(slice(None),) * axes.index(name) + (0,)]
            axes = [a for a in axes if a != name]

    level = da.moveaxis(level, axes.index("c"), -1)  # napari's rgb=True wants channel last
    return level.rechunk({2: -1})  # the three channels of a tile belong in one chunk


def read_pyramid(path: Path, scene: int = 0) -> tuple[SlideInfo, list[da.Array]]:
    """Return (info, zoom levels) for one scene, with level 0 at full resolution."""
    info = read_info(path, scene)
    series = image_series(path)[scene]
    ms = _multiscales(path, series)
    axes = [a["name"] for a in ms["axes"]]

    levels = [read_level(path, series, d["path"], axes) for d in ms["datasets"]]
    return info, levels
