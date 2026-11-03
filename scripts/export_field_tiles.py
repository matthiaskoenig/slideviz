"""Overlapping tiles over square fields of H&E slides, for a necrosis map finer than the tile grid.

    uv run python scripts/export_field_tiles.py --slides <dir> --out <dir> \
        --field mouse_apap_281mg_m1_he 4250 12750 [--size-um 1000] [--stride-um 15] \
        [--tile-um 75 --target-um-per-px 0.5]
    uv run python scripts/export_field_tiles.py --slides <dir> --out <dir> \
        --slide rat_mdo13_j25158_he_scan20251120

A slide is named by the stem of its sidecar in --slides (<stem>.json or <stem>.ome.json); the sidecar names
the image file and gives the slide's block, animal, species and dose (none for rat). A field is a slide and
its centre in level-0 px (row, col); --slide takes a whole slide, as the square around its tissue and a tile
beyond each side. Tiles are cut as for training (same size, level and tissue rule; --tile-um and
--target-um-per-px as given to export_tiles.py), every --stride-um.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from slideviz.analysis.tiling import (
    MIN_TISSUE,
    TARGET_UM_PER_PX,
    TILE_UM,
    coverage,
    pick_level_um,
)
from slideviz.analysis.tissue import mask_from_level, pick_level, pyramid_factor
from slideviz.data.registration import read_sidecar
from slideviz.io.reader import open_slide

SIZE_UM = 1000.0
STRIDE_UM = 15.0  # below one hepatocyte, so the outline is placed finer than a cell


def he_slide(slides: Path, slide: str) -> tuple[Path, dict]:
    """The image path and sidecar of the H&E slide with this stem; stops on another stain."""
    sidecar = read_sidecar(slides, slide)
    if sidecar["stain"] != "he":
        raise SystemExit(f"{slide} is stained {sidecar['stain']}; the necrosis model reads H&E")
    return slides / sidecar["file"], sidecar


def tissue_square(slides: Path, slide: str, tile_um: float) -> tuple[tuple[int, int], float]:
    """Centre in level-0 px and edge in um of the square around a slide's tissue, a tile beyond each side."""
    info, levels = open_slide(he_slide(slides, slide)[0])
    coarse = pick_level(levels)
    mask = mask_from_level(np.asarray(levels[coarse]))
    factor = pyramid_factor(levels[0].shape[1], levels[coarse].shape[1])
    rows, cols = np.nonzero(mask)
    top, bottom = rows.min() * factor - factor, (rows.max() + 1) * factor
    left, right = cols.min() * factor - factor, (cols.max() + 1) * factor
    size_um = max(bottom - top, right - left) * info.pixel_size_um + 2 * tile_um
    return (round((top + bottom) / 2), round((left + right) / 2)), size_um


def export_field(slides: Path, slide: str, centre: tuple[int, int], size_um: float, stride_um: float,
                 out: Path, tile_um: float = TILE_UM, target_um_per_px: float = TARGET_UM_PER_PX,
                 ) -> tuple[list[dict], dict]:
    """Write one field's overlapping tiles; return their manifest rows and the field's geometry."""
    path, sidecar = he_slide(slides, slide)
    info, levels = open_slide(path)
    level = pick_level_um([level.shape for level in levels], info.pixel_size_um, target_um_per_px)
    to_level = levels[0].shape[1] / levels[level].shape[1]
    um_per_px = info.pixel_size_um * to_level
    size_px = round(tile_um / um_per_px)
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

    (out / "tiles" / slide).mkdir(parents=True, exist_ok=True)
    rows = []
    digits = max(3, len(str(count - 1)))  # at least three, more once an index reaches 1000
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if y < top or x < left or y + size_px > bottom or x + size_px > right:
                continue
            tissue = coverage(mask, (height, width), int(y), int(x), size_px)
            if tissue < MIN_TISSUE:
                continue
            name = f"{slide}/{slide}_f{i:0{digits}d}_{j:0{digits}d}.png"
            Image.fromarray(region[y - top:y - top + size_px, x - left:x - left + size_px]).save(
                out / "tiles" / name)
            rows.append({
                "tile": name, "slide": slide, "block": sidecar["serial_block"], "animal": sidecar["animal_id"],
                "species": sidecar["species"], "dose_mg": sidecar.get("dose_mg_per_kg"), "row": i, "col": j, "y": int(y), "x": int(x), "level": level,
                "size_px": size_px, "tissue": round(tissue, 4),
                "necrosis": 0.0,  # unannotated; embed_tiles.py expects the field
            })

    geometry = {
        "slide": slide,
        "centre_level0_rc": list(centre),
        "size_um": size_um,
        "tile_um": tile_um,
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
    print(f"{slide}: {len(rows):,} of {count * count:,} positions hold tissue, "
          f"stride {geometry['stride_um']:.1f} um", flush=True)
    return rows, geometry


def main() -> None:
    """Export every requested field and write one manifest for the embedding stage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="directory of slides with their sidecars")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--field", nargs=3, action="append", default=[],
                        metavar=("SLIDE", "ROW", "COL"), help="slide stem and centre in level-0 px")
    parser.add_argument("--slide", action="append", default=[], metavar="SLIDE",
                        help="slide stem; the whole slide, as the square around its tissue")
    parser.add_argument("--size-um", type=float, default=SIZE_UM)
    parser.add_argument("--stride-um", type=float, default=STRIDE_UM)
    parser.add_argument("--tile-um", type=float, default=TILE_UM, help="as given to export_tiles.py")
    parser.add_argument("--target-um-per-px", type=float, default=TARGET_UM_PER_PX, help="as given to export_tiles.py")
    args = parser.parse_args()
    if not args.field and not args.slide:
        parser.error("give at least one --field or --slide")

    wanted = [(slide, (int(row), int(col)), args.size_um, False) for slide, row, col in args.field]
    wanted += [(slide, *tissue_square(args.slides, slide, args.tile_um), True) for slide in args.slide]
    rows, fields = [], {}
    for slide, centre, size_um, whole in wanted:
        found, geometry = export_field(args.slides, slide, centre, size_um, args.stride_um, args.out,
                                       args.tile_um, args.target_um_per_px)
        rows += found
        fields[slide] = {**geometry, "whole_slide": whole}

    manifest = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "slide_dir": str(args.slides),
        "n_tiles": len(rows),
        "slides": list(fields),
        "label": "necrosis",
        "label_is_fraction": True,
        "unlabelled": True,
        "tile_um": args.tile_um,
        "target_um_per_px": args.target_um_per_px,
        "min_tissue": MIN_TISSUE,
        "fields": fields,
        "tiles": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(rows):,} tiles from {len(fields)} fields to {args.out}")


if __name__ == "__main__":
    main()
