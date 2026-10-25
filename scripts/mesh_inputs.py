"""Cut one block's mesh inputs onto one H&E grid: tissue outline, lumina, necrosis map and CYP2E1 position.

Writes <animal>_mesh_inputs.npz, which the mesh env reads with numpy alone. With --centre a square field,
without it the whole section.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from skimage.measure import label

from slideviz.analysis.dab import NO_TISSUE, POSITION_SCALE
from slideviz.analysis.prediction import calls_on_level
from slideviz.analysis.tissue import (
    HOLE_PX,
    mask_from_level,
    on_level,
    pick_level,
    pyramid_factor,
)
from slideviz.data.provenance import code_version
from slideviz.data.registration import sha256
from slideviz.io.reader import open_slide

OUT_NAME = "{animal}_mesh_inputs.npz"


def grid_level(shapes: list[tuple[int, int]], shape: tuple[int, int]) -> int:
    """The H&E level a map was written on, found by its shape."""
    if shape not in shapes:
        raise SystemExit(f"a map of shape {shape} matches no H&E level {shapes}")
    return shapes.index(shape)


def window(shape: tuple[int, int], factor: int, centre: list[int] | None, size_um: float,
           um: float) -> tuple[int, int, int, int]:
    """Rows and cols (top, bottom, left, right) of the square around a level-0 centre, or the whole level."""
    if centre is None:
        return 0, shape[0], 0, shape[1]
    r, c = centre[0] // factor, centre[1] // factor
    half = int(size_um / 2 / um)
    return max(r - half, 0), min(r + half, shape[0]), max(c - half, 0), min(c + half, shape[1])


def summary(layers: dict, um: float) -> dict:
    """Shares and lumen counts of the cut layers, to check against earlier measurements."""
    measured = layers["measured"]
    lumina = np.bincount(label(layers["lumen"]).ravel())[1:]
    return {
        "shape": list(layers["tissue"].shape),
        "tissue_pct": round(100 * float(layers["tissue"].mean()), 2),
        "lumina": int(lumina.size),
        "largest_lumen_um2": round(float(lumina.max(initial=0) * um**2), 1),
        "necrotic_of_measured_pct": round(100 * float(layers["necrotic"].sum() / max(measured.sum(), 1)), 2),
        "cyp2e1_of_square_pct": round(100 * float((layers["cyp2e1"] != NO_TISSUE).mean()), 2),
        "cyp2e1_of_measured_pct": round(100 * float((layers["cyp2e1"] != NO_TISSUE)[measured].mean()), 2),
    }


def main() -> None:
    """Write the mesh inputs of one block and print their summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--he", type=Path, required=True, help="the block's H&E OME-TIFF")
    parser.add_argument("--lumen", type=Path, required=True, help="necrosis_lumen.py file of that H&E")
    parser.add_argument("--necrosis", type=Path, required=True, help="necrosis_slides.py call map of that H&E")
    parser.add_argument("--cyp2e1", type=Path, required=True, help="cyp2e1_readout.py map on that H&E")
    parser.add_argument("--centre", type=int, nargs=2, metavar=("ROW", "COL"),
                        help="field centre in H&E level-0 px; without it the whole section")
    parser.add_argument("--size-um", type=float, default=1000.0, help="field edge with --centre")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")

    stem = args.he.name.removesuffix(".ome.tiff")
    info, levels = open_slide(args.he)
    shapes = [tuple(level.shape[:2]) for level in levels]

    with np.load(args.lumen) as stored:
        lumen, lumen_slide = stored["lumen"], json.loads(str(stored["meta"]))["slide"]
    with np.load(args.cyp2e1) as stored:
        position, reference = stored["position"], str(stored["reference"])
    with np.load(args.necrosis) as stored:
        calls_meta = json.loads(str(stored["meta"]))  # the record only; calls_on_level reads the map
    for name, slide in (("lumen", lumen_slide), ("necrosis", calls_meta["slide"]), ("cyp2e1", reference)):
        if slide != stem:
            raise SystemExit(f"the {name} map belongs to {slide}, not {stem}")
    index = grid_level(shapes, lumen.shape)
    if position.shape != lumen.shape:
        raise SystemExit(f"cyp2e1 map {position.shape} and lumen mask {lumen.shape} are on different grids")

    factor = pyramid_factor(shapes[0][1], shapes[index][1])
    um = info.pixel_size_um * factor
    top, bottom, left, right = window(shapes[index], factor, args.centre, args.size_um, um)
    rows, cols = slice(top, bottom), slice(left, right)

    coarse = pick_level(levels)
    outline = mask_from_level(np.asarray(levels[coarse])[..., :3])
    outline_factor = pyramid_factor(shapes[index][1], shapes[coarse][1])
    necrotic, measured, _ = calls_on_level(args.necrosis, shapes, index)
    layers = {
        "tissue": on_level(outline, outline_factor, top, left, (bottom - top, right - left)),
        "lumen": lumen[rows, cols],
        "necrotic": necrotic[rows, cols],
        "measured": measured[rows, cols],
        "cyp2e1": position[rows, cols],
    }

    checked = summary(layers, um)
    meta = {
        "slide": stem,
        "animal": calls_meta["animal"],
        "level": index,
        "um_per_px": um,
        # pixel (r, c) is centred on (origin_um[0] + r * um_per_px, origin_um[1] + c * um_per_px) in the H&E frame
        "origin_um": [top * um, left * um],
        "window_level_rc": [top, bottom, left, right],
        "centre_level0_rc": args.centre,
        "size_um": args.size_um if args.centre else None,
        "layers": {
            "tissue": f"section outline, tissue.mask_from_level on H&E level {coarse}, holes under {HOLE_PX} px there filled",
            "lumen": "vessel lumina and tears inside the outline (necrosis_lumen.py)",
            "necrotic": "necrosis map, majority of its calls per pixel",
            "measured": "tissue the necrosis map covers (no glass, no lumen)",
            "cyp2e1": f"position between the local low (0) and high ({POSITION_SCALE}) DAB, {NO_TISSUE} no value",
        },
        "sources": {
            "he": str(args.he),
            **{name: {"file": str(path), "sha256": sha256(path)}
               for name, path in (("lumen", args.lumen), ("necrosis", args.necrosis), ("cyp2e1", args.cyp2e1))},
        },
        "necrosis_cutoff": calls_meta.get("cutoff"),
        "summary": checked,
        "code": code_version(Path(__file__)),
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / OUT_NAME.format(animal=meta["animal"])
    np.savez_compressed(path, **layers, meta=json.dumps(meta))
    print(f"{path.name}: {json.dumps(checked)}")


if __name__ == "__main__":
    main()
