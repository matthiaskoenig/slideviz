"""Report what is inside a slide file, without decoding any image data.

    uv run python scripts/probe_slide.py /path/to/dir      # every slide in it, e.g. uv run python scripts/probe_slide.py /home/michelle/Projects/image-analysis/APAP
"""

from __future__ import annotations

import sys
from pathlib import Path

# Whole-slide formats, by the scanner vendor that writes them
SUFFIXES = (
    ".czi",                    # Zeiss
    ".ome.tiff", ".ome.tif",   # OME-TIFF, vendor-neutral
    ".tiff", ".tif",           # plain TIFF
    ".ndpi",                   # Hamamatsu NanoZoomer
    ".svs",                    # Leica / Aperio
    ".qptiff",                 # Akoya / PerkinElmer
)

# Zeiss labels
DIM_NAMES = {
    "X": "width",
    "Y": "height",
    "Z": "focal planes",
    "T": "timepoints",
    "C": "channels",
    "B": "blocks",
}


def describe_dims(box: dict[str, tuple[int, int]]) -> list[str]:
    """Turn raw bounding box into one readable line per axis.

    Each entry is a half-open range (start, stop), so its length is stop - start.
    """
    lines = []
    for code, (start, stop) in box.items():
        name = DIM_NAMES.get(code, "?")
        extent = stop - start
        # X and Y also get start and stop, other axes are only counts
        if code in ("X", "Y"):
            lines.append(f"    {code} {name:<12} {extent:>7} px   from {start} to {stop}")
        else:
            lines.append(f"    {code} {name:<12} {extent:>7}")
    return lines


def probe_czi(path: Path) -> None:
    """Report a Zeiss .czi: scenes, extent, channels, pixel size."""
    import pylibCZIrw.czi as pyczi

    # Reads the header only, no image data
    with pyczi.open_czi(str(path)) as czi:
        box = czi.total_bounding_box
        scenes = czi.scenes_bounding_rectangle

        print("  dimensions  (extent of each axis the file uses)")
        for line in describe_dims(box):
            print(line)

        # Bgr24 = 8 bits each of blue, green, red (in that order)
        print(f"  pixel type  {czi.pixel_types}")

        # A scene is one independently scanned region on the glass
        print(f"  scenes      {len(scenes)}")
        for i, rect in scenes.items():
            print(f"    [{i}] x={rect.x} y={rect.y} w={rect.w} h={rect.h}")

        # The origin is a stage position -> size is the difference
        x0, x1 = box["X"]
        y0, y1 = box["Y"]
        w, h = x1 - x0, y1 - y0
        print(f"  size        {w} x {h} px  ({w * h / 1e9:.2f} Gpx)")
        if x0 < 0 or y0 < 0:
            print(f"  origin      ({x0}, {y0})")

        # Compressed on disk against decompressed in RAM
        print(f"  on disk     {path.stat().st_size / 1e6:.0f} MB")
        print(f"  RGB in RAM  {w * h * 3 / 1e9:.1f} GB")

        # Pixel size: under Scaling in metadata, in metres
        try:
            distances = czi.metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
            for d in distances:
                if d["@Id"] in ("X", "Y"):
                    print(f"  pixel {d['@Id']}     {float(d['Value']) * 1e6:.4f} um")
        except (KeyError, TypeError):
            print("  pixel size  not found in metadata, image has no physical scale")

        if len(scenes) > 1:
            print("  NOTE: multi-scene file, readers assuming one scene will miss data")


def probe_tiff(path: Path) -> None:
    """Report a TIFF-based slide: series, pyramid levels, tiling.
    """
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        print(f"  format      {'OME-TIFF' if tif.is_ome else 'TIFF'}")
        print(f"  on disk     {path.stat().st_size / 1e6:.0f} MB")

        # A series is one image, its levels are stored zoomed-out copies
        print(f"  series      {len(tif.series)}")
        for i, series in enumerate(tif.series):
            levels = getattr(series, "levels", [series])
            print(f"    [{i}] {series.shape} {series.dtype} axes={series.axes} "
                  f"pyramid levels={len(levels)}")

        # Tiled means the image is stored in squares
        page = tif.pages[0]
        print(f"  compression {page.compression.name}")
        print(f"  tiled       {page.is_tiled}"
              f"{f' ({page.tilewidth}x{page.tilelength})' if page.is_tiled else ''}")

        # Either of these makes a readable file slow to view
        if not page.is_tiled:
            print("  NOTE: not tiled, reading a region costs whole image rows")
        if len(getattr(tif.series[0], "levels", [1])) == 1:
            print("  NOTE: no pyramid, zoomed-out views must read full resolution")


def probe(path: Path) -> None:
    """Print a report for one file, dispatching on its extension."""
    print(f"\n{path.name}")
    try:
        if path.suffix.lower() == ".czi":
            probe_czi(path)
        else:
            probe_tiff(path)
    # pylibCZIrw raises RuntimeError, tifffile raises a ValueError subclass, unreadable files raise OSError; report and carry on
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")


def main() -> None:
    """Probe every path given; directories expand to the slides inside them."""
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        sys.exit(__doc__)

    paths: list[Path] = []
    for arg in args:
        if arg.is_dir():
            paths += sorted(p for p in arg.iterdir() if p.suffix.lower() in SUFFIXES)
        else:
            paths.append(arg)

    for path in paths:
        probe(path)


if __name__ == "__main__":
    main()
