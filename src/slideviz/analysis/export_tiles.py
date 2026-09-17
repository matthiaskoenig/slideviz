"""Annotated tiles written to disk as images, ready for an encoder in another environment.

Tiling and labelling need the slide readers, embedding needs CUDA wheels, and the two
environments are deliberately separate. So this stage ends at a directory of PNGs plus
one manifest, which the embedding stage reads without importing slideviz.

The manifest carries the animal each tile came from, because the split is by animal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from slideviz.analysis.annotation import cvat_polygons, label_grid, read_overview
from slideviz.analysis.tiling import build_grid, read_tile
from slideviz.io.reader import open_slide

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Row:
    """One written tile: where it came from, and what it is labelled."""

    tile: str  # filename, relative to the tile directory
    slide: str
    animal: str  # the split is by animal, so it travels with every row
    dose_mg: int
    row: int
    col: int
    y: int  # top-left pixel on `level`
    x: int
    level: int
    size_px: int
    tissue: float
    necrosis: float  # fraction of the tile inside an annotation polygon


def animal_of(slide: str) -> str:
    """The animal a slide name belongs to, as dose plus individual."""
    parts = slide.split("_")
    return f"{parts[2]}_{parts[3]}"  # mouse_apap_281mg_m1_he -> 281mg_m1


def dose_of(slide: str) -> int:
    """Dose in mg/kg, read from the slide name."""
    return int(slide.split("_")[2].removesuffix("mg"))


def export_slide(
    slide_path: Path,
    polygons: list[np.ndarray],
    overview: dict,
    out_dir: Path,
    scene: int = 0,
) -> list[Row]:
    """Write every tissue tile of one slide as a PNG, with its annotated fraction."""
    grid = build_grid(slide_path, scene)
    labels = label_grid(grid, polygons, overview)
    _, levels = open_slide(slide_path, scene)

    slide = slide_path.name.split(".")[0]
    animal, dose = animal_of(slide), dose_of(slide)
    tile_dir = out_dir / "tiles" / animal
    tile_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for coverage, tile in zip(labels.tiles, grid.tiles, strict=True):
        name = f"{animal}_r{coverage.row:04d}_c{coverage.col:04d}.png"
        Image.fromarray(read_tile(levels, tile)).save(tile_dir / name)
        rows.append(
            Row(
                tile=f"{animal}/{name}",
                slide=slide,
                animal=animal,
                dose_mg=dose,
                row=coverage.row,
                col=coverage.col,
                y=coverage.y,
                x=coverage.x,
                level=grid.level,
                size_px=coverage.size_px,
                tissue=round(coverage.tissue, 4),
                necrosis=round(coverage.necrosis, 4),
            )
        )

    log.info("%s: %d tiles written to %s", slide, len(rows), tile_dir)
    return rows


def export_annotated(
    annotations: Path, slide_dir: Path, overview_dir: Path, out_dir: Path
) -> dict:
    """Write tiles for every slide in the annotation export, plus one manifest."""
    export = json.loads(annotations.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for key, record in export.items():
        stem = key.removesuffix(".png")
        slide = stem.removesuffix("_s0")
        overview = read_overview(overview_dir / f"{stem}.json")
        if (overview["width"], overview["height"]) != (record["width"], record["height"]):
            raise ValueError(
                f"{stem}: CVAT saw {record['width']}x{record['height']}, "
                f"sidecar records {overview['width']}x{overview['height']}"
            )
        polygons = [cvat_polygons(p["points"]) for p in record["polygons"]]
        rows += export_slide(
            slide_dir / f"{slide}.ome.tiff", polygons, overview, out_dir
        )

    necrosis = np.array([r.necrosis for r in rows])
    manifest = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "annotations": str(annotations),
        "slide_dir": str(slide_dir),
        "n_tiles": len(rows),
        "animals": sorted({r.animal for r in rows}),
        "label": "necrosis",
        "label_is_fraction": True,  # a threshold is the training stage's choice, not this one
        "necrotic_over_half": int((necrosis > 0.5).sum()),
        "clean": int((necrosis == 0).sum()),
        "boundary_0.1_to_0.9": int(((necrosis > 0.1) & (necrosis < 0.9)).sum()),
        "tiles": [asdict(r) for r in rows],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("%d tiles across %d animals", len(rows), len(manifest["animals"]))
    return manifest
