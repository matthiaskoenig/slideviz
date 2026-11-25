"""Pick a reader backend by file suffix; a new format is a new module plus an entry in BACKENDS."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import dask.array as da

from slideviz.io.czi import SlideInfo, read_pyramid
from slideviz.io.ndpi import read_pyramid as read_ndpi_pyramid
from slideviz.io.ome_tiff import read_pyramid as read_ome_tiff_pyramid
from slideviz.io.ome_zarr import read_pyramid as read_zarr_pyramid

# Suffix to the function that reads it; one entry per supported format
BACKENDS = {
    ".czi": read_pyramid,
    ".zarr": read_zarr_pyramid,
    ".ndpi": read_ndpi_pyramid,
    ".ome.tiff": read_ome_tiff_pyramid,
}


def suffix_of(path: Path) -> str:
    """Longest BACKENDS suffix the lowercased name ends with, else its last suffix."""
    name = path.name.lower()
    matches = [s for s in BACKENDS if name.endswith(s)]
    return max(matches, key=len) if matches else path.suffix.lower()


def slide_stem(path: Path) -> str:
    """The file name without its slide format suffix."""
    suffix = suffix_of(path)
    return path.name[: -len(suffix)] if suffix in BACKENDS and path.name.lower().endswith(suffix) else path.stem


def find_slide(directory: Path, stem: str) -> Path:
    """The slide `<stem><suffix>` in `directory`, whichever supported format it has; more than one match is an error."""
    found = [directory / f"{stem}{suffix}" for suffix in BACKENDS if (directory / f"{stem}{suffix}").exists()]
    if not found:
        raise FileNotFoundError(f"no slide {stem} in {directory}; known formats: {', '.join(sorted(BACKENDS))}")
    if len(found) > 1:
        raise ValueError(f"slide {stem} is ambiguous in {directory}: {', '.join(p.name for p in found)}")
    return found[0]


def slide_paths(directory: Path, pattern: str = "*") -> list[Path]:
    """Slides in `directory` whose stem matches the glob `pattern`, one per stem, sorted by stem."""
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        stem = slide_stem(path)
        if suffix_of(path) in BACKENDS and fnmatch(stem, pattern):
            found.setdefault(stem, find_slide(directory, stem))
    return [found[stem] for stem in sorted(found)]


def open_slide(path: Path, scene: int = 0) -> tuple[SlideInfo, list[da.Array]]:
    """Read a slide as (info, zoom levels), dispatching on its suffix."""
    suffix = suffix_of(path)
    if suffix not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"no reader for '{suffix}': {path.name}. Known formats: {known}")
    return BACKENDS[suffix](path, scene)
