"""Pick a reader backend by file suffix; a new format is a new module plus an entry in BACKENDS."""

from __future__ import annotations

from pathlib import Path

import dask.array as da

from slideviz.io.czi import SlideInfo, read_info, read_pyramid
from slideviz.io.ndpi import read_info as read_ndpi_info
from slideviz.io.ndpi import read_pyramid as read_ndpi_pyramid
from slideviz.io.ome_tiff import read_info as read_ome_tiff_info
from slideviz.io.ome_tiff import read_pyramid as read_ome_tiff_pyramid
from slideviz.io.ome_zarr import read_info as read_zarr_info
from slideviz.io.ome_zarr import read_pyramid as read_zarr_pyramid

# Suffix to the function that reads it; one entry per supported format
BACKENDS = {
    ".czi": read_pyramid,
    ".zarr": read_zarr_pyramid,
    ".ndpi": read_ndpi_pyramid,
    ".ome.tiff": read_ome_tiff_pyramid,
}

# Suffix to the function that reads its header only; the same keys as BACKENDS
INFO = {
    ".czi": read_info,
    ".zarr": read_zarr_info,
    ".ndpi": read_ndpi_info,
    ".ome.tiff": read_ome_tiff_info,
}


def suffix_of(path: Path) -> str:
    """Longest BACKENDS suffix the lowercased name ends with, else its last suffix."""
    name = path.name.lower()
    matches = [s for s in BACKENDS if name.endswith(s)]
    return max(matches, key=len) if matches else path.suffix.lower()


def _known(path: Path) -> str:
    """The path's BACKENDS suffix, raising for a format without a reader."""
    suffix = suffix_of(path)
    if suffix not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"no reader for '{suffix}': {path.name}. Known formats: {known}")
    return suffix


def open_slide(path: Path, scene: int = 0) -> tuple[SlideInfo, list[da.Array]]:
    """Read a slide as (info, zoom levels), dispatching on its suffix."""
    return BACKENDS[_known(path)](path, scene)


def slide_info(path: Path, scene: int = 0) -> SlideInfo:
    """Read a slide's geometry and pixel size from its header, without decoding pixels."""
    return INFO[_known(path)](path, scene)
