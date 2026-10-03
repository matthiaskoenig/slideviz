"""Overlapping tiles over square fields of H&E slides, for a necrosis map finer than the tile grid.

    uv run python scripts/export_field_tiles.py --slides <APAP_tiff> --out <dir> \
        --field 281mg_m1 4250 12750 --field 500mg_m1 4250 19890 [--size-um 1000] [--stride-um 15]

A field is an animal and its centre in H&E level-0 px (row, col). Tiles are cut as for
training (same size, level and tissue rule) but every --stride-um; the manifest reads
like export_tiles.py's, so ml/embed_tiles.py embeds it unchanged.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from slideviz.analysis.export_tiles import dose_of
from slideviz.analysis.tiling import MIN_TISSUE, TILE_UM, coverage, pick_level_um
from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.io.reader import open_slide

SIZE_UM = 1000.0
STRIDE_UM = 15.0  # below one hepatocyte, so the outline is placed finer than a cell


def export_field(slides: Path, animal: str, centre: tuple[int, int], size_um: float,
                 stride_um: float, out: Path) -> tuple[list[dict], dict]:
    """Write one field's overlapping tiles; return their manifest rows and the field's geometry."""
    path = slides / f"mouse_apap_{animal}_he.ome.tiff"
    info, levels = open_slide(path)
    level = pick_level_um(levels, info.pixel_size_um)
    to_level = levels[0].shape[1] / levels[level].shape[1]
    um_per_px = info.pixel_size_um * to_level
    size_px = round(TILE_UM / um_per_px)
    stride_px = max(1, round(stride_um / um_per_px))
    height, width = levels[level].shape[:2]
    mask = mask_from_level(np.asarray(levels[pick_level(levels)]))

    # tile centres on a stride grid across the field, then each tile's top-left corner
    half = size_um / 2 / um_per_px
    count = int(2 * half // stride_px) + 1
    steps = np.arange(count) * stride_px
    ys = np.round(centre[0] / to_level - half + steps - size_px / 2).astype(int)
    xs = np.round(centre[1] / to_level - half + steps - size_px / 2).astype(int)

    # one read for the field and its tile-wide border, cut into tiles in memory
    top, left = max(int(ys[0]), 0), max(int(xs[0]), 0)
    bottom, right = min(int(ys[-1]) + size_px, height), min(int(xs[-1]) + size_px, width)
    region = np.asarray(levels[level][top:bottom, left:right])

    slide = path.name.split(".")[0]
    (out / "tiles" / animal).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if y < top or x < left or y + size_px > bottom or x + size_px > right:
                continue
            tissue = coverage(mask, (height, width), int(y), int(x), size_px)
            if tissue < MIN_TISSUE:
                continue
            name = f"{animal}/{animal}_f{i:03d}_{j:03d}.png"
            Image.fromarray(region[y - top:y - top + size_px, x - left:x - left + size_px]).save(
                out / "tiles" / name)
            rows.append({
                "tile": name, "slide": slide, "animal": animal, "dose_mg": dose_of(slide),
                "row": i, "col": j, "y": int(y), "x": int(x), "level": level,
                "size_px": size_px, "tissue": round(tissue, 4),
                "necrosis": 0.0,  # unannotated; embed_tiles.py expects the field
            })

    geometry = {
        "slide": slide,
        "centre_level0_rc": list(centre),
        "size_um": size_um,
        "level": level,
        "um_per_px": um_per_px,
        "size_px": size_px,
        "stride_px": stride_px,
        "stride_um": stride_px * um_per_px,
        "grid_shape": [count, count],
        # level pixel of the first tile centre, row and column
        "first_centre": [float(ys[0] + size_px / 2), float(xs[0] + size_px / 2)],
        "n_tiles": len(rows),
    }
    print(f"{animal}: {len(rows):,} of {count * count:,} positions hold tissue, "
          f"stride {geometry['stride_um']:.1f} um", flush=True)
    return rows, geometry


def main() -> None:
    """Export every requested field and write one manifest for the embedding stage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="directory of OME-TIFFs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--field", nargs=3, action="append", required=True,
                        metavar=("ANIMAL", "ROW", "COL"), help="centre in H&E level-0 px")
    parser.add_argument("--size-um", type=float, default=SIZE_UM)
    parser.add_argument("--stride-um", type=float, default=STRIDE_UM)
    args = parser.parse_args()

    rows, fields = [], {}
    for animal, row, col in args.field:
        found, geometry = export_field(args.slides, animal, (int(row), int(col)),
                                       args.size_um, args.stride_um, args.out)
        rows += found
        fields[animal] = geometry

    manifest = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "slide_dir": str(args.slides),
        "n_tiles": len(rows),
        "animals": list(fields),
        "label": "necrosis",
        "label_is_fraction": True,
        "unlabelled": True,
        "tile_um": TILE_UM,
        "min_tissue": MIN_TISSUE,
        "fields": fields,
        "tiles": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(rows):,} tiles from {len(fields)} fields to {args.out}")


if __name__ == "__main__":
    main()
