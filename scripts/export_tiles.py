"""Write annotated tiles and their manifest for the embedding stage.

    uv run python scripts/export_tiles.py --annotations <cvat_polygons.json> \
        --slides <APAP_tiff> --overviews <annotation_overviews> --out <tiles>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from slideviz.analysis.export_tiles import export_annotated
from slideviz.analysis.tiling import TARGET_UM_PER_PX, TILE_UM


def main() -> None:
    """Export every annotated slide's tiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="CVAT polygon export")
    parser.add_argument("--slides", type=Path, required=True, help="directory of OME-TIFFs")
    parser.add_argument("--overviews", type=Path, required=True, help="overview PNG sidecars")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confirmed-empty", type=Path, default=None,
                        help="slides confirmed to hold no necrosis")
    parser.add_argument("--rewrite", action="store_true", help="cut every tile again")
    parser.add_argument("--tile-um", type=float, default=TILE_UM)
    parser.add_argument("--target-um-per-px", type=float, default=TARGET_UM_PER_PX)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = export_annotated(
        args.annotations, args.slides, args.overviews, args.out, args.confirmed_empty,
        rewrite=args.rewrite, tile_um=args.tile_um,
        target_um_per_px=args.target_um_per_px,
    )

    print(f"\n=== {manifest['n_tiles']:,} tiles to {args.out} ===")
    print(f"  grid              {manifest['tile_um']} um, "
          f"{manifest['size_px']} px at level {manifest['level']}")
    print(f"  animals           {', '.join(manifest['animals'])}")
    print(f"  necrotic > 0.5    {manifest['necrotic_over_half']:,}")
    print(f"  clean, exactly 0  {manifest['clean']:,}")
    print(f"  boundary          {manifest['boundary_0.1_to_0.9']:,}")


if __name__ == "__main__":
    main()
