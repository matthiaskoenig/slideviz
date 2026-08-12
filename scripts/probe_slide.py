"""Report what is inside a slide file.

    uv run python scripts/probe_slide.py <file> [<file> ...]
    uv run python scripts/probe_slide.py /path/to/dir      # every slide in it

Answers: how many scenes, how big, what pixel type, how many channels, and what
one pixel is worth in micrometres.
"""

from __future__ import annotations

import sys
from pathlib import Path

SUFFIXES = (".czi", ".ome.tiff", ".ome.tif", ".tiff", ".tif", ".ndpi", ".svs", ".qptiff")


def probe_czi(path: Path) -> None:
    import pylibCZIrw.czi as pyczi

    with pyczi.open_czi(str(path)) as czi:
        box = czi.total_bounding_box
        scenes = czi.scenes_bounding_rectangle

        print(f"  dims        {dict(box)}")
        print(f"  pixel types {czi.pixel_types}")
        print(f"  scenes      {len(scenes)}")
        for i, rect in scenes.items():
            print(f"    [{i}] x={rect.x} y={rect.y} w={rect.w} h={rect.h}")

        x0, x1 = box["X"]
        y0, y1 = box["Y"]
        w, h = x1 - x0, y1 - y0
        c0, c1 = box.get("C", (0, 1))
        print(f"  size        {w} x {h} px  ({w * h / 1e9:.2f} Gpx)")
        print(f"  channels    {c1 - c0}")
        print(f"  origin      ({x0}, {y0}){'  <- negative, stage-relative' if x0 < 0 or y0 < 0 else ''}")
        print(f"  RGB in RAM  {w * h * 3 / 1e9:.1f} GB")

        try:
            distances = czi.metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
            for d in distances:
                if d["@Id"] in ("X", "Y"):
                    print(f"  pixel {d['@Id']}     {float(d['Value']) * 1e6:.4f} um")
        except (KeyError, TypeError):
            print("  pixel size  not found in metadata")

        if len(scenes) > 1:
            print("  NOTE: multi-scene file, readers assuming one scene will be wrong")


def probe_tiff(path: Path) -> None:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        print(f"  format      {'OME-TIFF' if tif.is_ome else 'TIFF'}")
        print(f"  series      {len(tif.series)}")
        for i, series in enumerate(tif.series):
            levels = getattr(series, "levels", [series])
            print(f"    [{i}] {series.shape} {series.dtype} axes={series.axes} "
                  f"pyramid levels={len(levels)}")
        page = tif.pages[0]
        print(f"  compression {page.compression.name}")
        print(f"  tiled       {page.is_tiled}"
              f"{f' ({page.tilewidth}x{page.tilelength})' if page.is_tiled else ''}")
        if not page.is_tiled:
            print("  NOTE: not tiled, random access will be slow")
        if len(getattr(tif.series[0], "levels", [1])) == 1:
            print("  NOTE: no pyramid, zoomed-out views must read full resolution")


def probe(path: Path) -> None:
    print(f"\n{path.name}")
    print(f"  {path.stat().st_size / 1e6:.0f} MB")
    try:
        if path.suffix.lower() == ".czi":
            probe_czi(path)
        else:
            probe_tiff(path)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")


def main() -> None:
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
