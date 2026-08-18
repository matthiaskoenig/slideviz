"""Pick a reader backend by file suffix.

Keeps callers from importing a format-specific reader, so a new format
is a new backend and an entry below.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da

from slideviz.czi import SlideInfo, read_pyramid

# Suffix to the function that reads it; one entry per supported format
BACKENDS = {
    ".czi": read_pyramid,
}


def suffix_of(path: Path) -> str:
    """Longest known suffix of a path, so .ome.tiff wins over .tiff."""
    name = path.name.lower()
    matches = [s for s in BACKENDS if name.endswith(s)]
    return max(matches, key=len) if matches else path.suffix.lower()


def open_slide(path: Path, scene: int = 0) -> tuple[SlideInfo, list[da.Array]]:
    """Read a slide as (info, zoom levels), dispatching on its suffix."""
    suffix = suffix_of(path)
    if suffix not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"no reader for '{suffix}': {path.name}. Known formats: {known}")
    return BACKENDS[suffix](path)
