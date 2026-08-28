"""Open a registered pair in napari, aligned, from the sidecars.

    uv run python scripts/view_registered.py <slide_dir> <stem-of-moving-slide>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import napari

from slideviz.reader import open_slide
from slideviz.registration import napari_affine
from slideviz.schema import Slide


def add(viewer, path: Path, colormap: str, affine=None) -> None:
    """One slide as a lazy multiscale layer, optionally transformed."""
    info, levels = open_slide(path)
    viewer.add_image(
        levels,
        name=path.stem,
        rgb=True,
        multiscale=True,
        scale=(info.pixel_size_um, info.pixel_size_um),
        units="um",
        affine=affine,
        opacity=0.7,
        blending="additive",  # so the two stains show through each other
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_dir", type=Path)
    parser.add_argument("moving", help="file stem of the registered slide")
    args = parser.parse_args()

    sidecar = args.slide_dir / f"{args.moving}.json"
    slide = Slide(**json.loads(sidecar.read_text()))
    if slide.registration is None:
        raise SystemExit(f"{args.moving} has no registration in its sidecar")

    reference = slide.registration.reference

    def slide_file(stem: str) -> Path:
        """The image with this stem, not the sidecar that shares it."""
        return next(
            p for p in args.slide_dir.iterdir()
            if p.stem == stem and p.suffix != ".json"
        )

    fixed_path, moving_path = slide_file(reference), slide_file(args.moving)

    viewer = napari.Viewer(title=f"{reference}  +  {args.moving}")
    info = add(viewer, fixed_path, "green")
    add(viewer, moving_path, "magenta", affine=napari_affine(slide.registration, info.pixel_size_um))

    print(f"reference {reference}")
    print(f"moving    {args.moving}  ({slide.registration.error_um} um, {slide.registration.method})")
    napari.run()


if __name__ == "__main__":
    main()
