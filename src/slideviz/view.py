"""Open .czi slides in napari.

    uv run slideviz-view --pair "0mg m1"
    uv run slideviz-view "<path>.czi" ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slideviz.czi import read_pyramid

APAP_DIR = Path("/home/michelle/Projects/image-analysis/APAP")


def find_pair(spec: str, directory: Path = APAP_DIR) -> list[Path]:
    """Find both stain sections for one animal, e.g. spec="0mg m1"."""
    wanted = spec.split()
    matches = sorted(
        p for p in directory.glob("*.czi")
        # filenames contain literal quotes, so split off the tokens we compare
        if all(w in p.stem.strip("'").split() for w in wanted)
    )
    if not matches:
        raise SystemExit(f"no slides matching {spec!r} in {directory}")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--pair", help="animal spec, e.g. '158mg m1'")
    args = parser.parse_args()

    paths = find_pair(args.pair) if args.pair else args.paths
    if not paths:
        parser.error("give at least one .czi path, or --pair")

    import napari  # slow: pulls in Qt

    viewer = napari.Viewer(title="slideviz")

    for path in paths:
        info, levels = read_pyramid(path)
        print(f"{path.name}  {info.width}x{info.height} px  {info.pixel_size_um:.4f} um/px")
        viewer.add_image(
            levels,
            name=path.stem.strip("'"),
            rgb=True,
            multiscale=True,
            scale=(info.pixel_size_um, info.pixel_size_um),
        )

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    napari.run()


if __name__ == "__main__":
    main()
