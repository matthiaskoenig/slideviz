"""A downscaled slide PNG for CVAT annotation, with a JSON sidecar of the level and pixel size that map drawings back to the slide."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from slideviz.io.reader import open_slide

log = logging.getLogger(__name__)

# CVAT handles this comfortably and it still resolves the necrosis front
MAX_EDGE_PX = 4000


@dataclass(frozen=True)
class Overview:
    """One exported image and how it relates to the slide it came from."""

    slide: str
    scene: int
    image: str
    level: int
    height: int
    width: int
    um_per_px: float
    scale_to_level0: float  # multiply an overview coordinate by this to reach level 0
    exported: str


def pick_export_level(levels: list, max_edge_px: int = MAX_EDGE_PX) -> int:
    """Index of the finest level whose longest edge fits within `max_edge_px`."""
    for i, level in enumerate(levels):
        if max(level.shape[0], level.shape[1]) <= max_edge_px:
            return i
    return len(levels) - 1  # coarsest level when none fits


def export_overview(
    path: Path, out_dir: Path, scene: int = 0, max_edge_px: int = MAX_EDGE_PX
) -> Overview:
    """Write one slide as a PNG for annotation, plus the JSON that locates it."""
    info, levels = open_slide(path, scene)
    level = pick_export_level(levels, max_edge_px)

    image = np.asarray(levels[level])
    height, width = image.shape[:2]
    scale = levels[0].shape[1] / width

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{path.name.split('.')[0]}_s{scene}"
    image_path = out_dir / f"{stem}.png"
    Image.fromarray(image).save(image_path)

    overview = Overview(
        slide=path.name,
        scene=scene,
        image=image_path.name,
        level=level,
        height=height,
        width=width,
        um_per_px=info.pixel_size_um * scale,
        scale_to_level0=scale,
        exported=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (out_dir / f"{stem}.json").write_text(json.dumps(asdict(overview), indent=2) + "\n")

    log.info(
        "%s scene %d: level %d, %dx%d px, %.1f um/px",
        path.name,
        scene,
        level,
        width,
        height,
        overview.um_per_px,
    )
    return overview


def main() -> None:
    """Export overviews for annotation from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slides", type=Path, nargs="+", help="slide files")
    parser.add_argument("--out", type=Path, required=True, help="folder for the PNGs and their JSON")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--max-edge-px", type=int, default=MAX_EDGE_PX)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in args.slides:
        export_overview(path, args.out, args.scene, args.max_edge_px)


if __name__ == "__main__":
    main()
