"""Lazy tiled reading of Zeiss .czi whole-slide images.

Returns a list of dask arrays (one per zoom level) that
napari can consume directly as a multiscale image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pylibCZIrw.czi as pyczi

# Tile edge in level-0 pixels. ~12 MB per tile as RGB.
TILE = 2048


@dataclass(frozen=True)
class SlideInfo:
    path: Path
    x0: int  # stage-relative origin, usually negative
    y0: int
    width: int
    height: int
    pixel_size_um: float


def read_info(path: Path) -> SlideInfo:
    """Read geometry and scale without decoding any image data."""
    with pyczi.open_czi(str(path)) as czi:
        box = czi.total_bounding_box
        x0, x1 = box["X"]
        y0, y1 = box["Y"]

        distances = czi.metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
        pixel_size_m = next(float(d["Value"]) for d in distances if d["@Id"] == "X")

        return SlideInfo(
            path=path,
            x0=x0,
            y0=y0,
            width=x1 - x0,
            height=y1 - y0,
            pixel_size_um=pixel_size_m * 1e6,
        )


def _read_tile(
    path: str, x: int, y: int, w: int, h: int, downsample: int, out_h: int, out_w: int
) -> np.ndarray:
    """Read one tile as RGB, shaped exactly (out_h, out_w, 3).

    The requested read is clipped to the slide edge, so it can come back
    smaller than the tile it has to fill; the remainder is padded white.
    """
    with pyczi.open_czi(path) as czi:
        # zoom does the downsampling inside the reader, coarse level
        # never touches full-resolution pixels
        tile = czi.read(
            plane={"T": 0, "Z": 0, "C": 0},
            roi=(x, y, w, h),
            zoom=1 / downsample,
            background_pixel=(255, 255, 255),
        )

    tile = np.asarray(tile)[..., ::-1]  # pylibCZIrw returns BGR

    if tile.shape[:2] != (out_h, out_w):
        padded = np.full((out_h, out_w, 3), 255, dtype=tile.dtype)
        padded[: tile.shape[0], : tile.shape[1]] = tile[:out_h, :out_w]
        tile = padded

    return tile


def read_level(info: SlideInfo, downsample: int) -> da.Array:
    """Build one zoom level as a lazy tiled dask array."""
    level_h = -(-info.height // downsample)
    level_w = -(-info.width // downsample)

    rows = []
    for y in range(0, level_h, TILE):
        row = []
        for x in range(0, level_w, TILE):
            th = min(TILE, level_h - y)
            tw = min(TILE, level_w - x)

            block = da.from_delayed(
                dask.delayed(_read_tile)(
                    str(info.path),
                    info.x0 + x * downsample,
                    info.y0 + y * downsample,
                    min(tw * downsample, info.width - x * downsample),
                    min(th * downsample, info.height - y * downsample),
                    downsample,
                    th,
                    tw,
                ),
                shape=(th, tw, 3),
                dtype=np.uint8,
            )
            row.append(block)
        rows.append(da.concatenate(row, axis=1))

    return da.concatenate(rows, axis=0)


def read_pyramid(path: Path, n_levels: int = 5) -> tuple[SlideInfo, list[da.Array]]:
    """Return (info, zoom levels) with level 0 at full resolution."""
    info = read_info(path)
    return info, [read_level(info, 2**i) for i in range(n_levels)]
