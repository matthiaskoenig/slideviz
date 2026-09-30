"""Refine slide alignment with DeepFlow and save the displacement field and summary.

    uv run python nonrigid_register.py 281mg_m2 375mg_m3 \
        --slides /data/michelle/slides --out /data/michelle/c1/fields
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from outline_register import _warp, read_level, shared_downsample, tissue_mask

EDGE_PX = 2048  # RAFT tore the tissue at this size, DeepFlow matched its 1024 px result
MIN_AGREEMENT = 0.99  # own resampling against VALIS's warped image


def um_per_px(path: Path) -> float:
    """Full-resolution pixel size from the TIFF resolution tag."""
    import pyvips

    image = pyvips.Image.new_from_file(str(path), page=0)
    return 1000.0 / image.xres  # vips stores pixels per mm


def sidecar_matrix(slides: Path, slide: str) -> np.ndarray:
    """The moving slide's matrix onto its reference, from its sidecar."""
    registration = json.loads((slides / f"{slide}.json").read_text())["registration"]
    return np.array(registration["matrix"], float)


def scaled(matrix: np.ndarray, downsample: float) -> np.ndarray:
    """The same transform in a grid shrunk by `downsample`."""
    shrink = np.diag([1 / downsample, 1 / downsample, 1.0])
    return shrink @ matrix @ np.linalg.inv(shrink)


def density(grey: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tissue bright and background zero, as uint8, the input DeepFlow gets."""
    return np.where(mask, 255 - grey, 0).astype(np.uint8)


def agreement(moving: np.ndarray, warped: np.ndarray, dxdy: np.ndarray, mask) -> float:
    """Correlation between VALIS's warped image and sampling `moving` at p + d."""
    from scipy.ndimage import map_coordinates

    rows, cols = np.mgrid[: moving.shape[0], : moving.shape[1]].astype(float)
    mine = map_coordinates(moving.astype(float), [rows + dxdy[1], cols + dxdy[0]], order=1)
    return float(np.corrcoef(mine[mask], np.asarray(warped, float)[mask])[0, 1])


def register_block(block: str, slides: Path, out: Path, edge: int) -> dict:
    """Run DeepFlow on one block and write its field and summary."""
    import valis
    from valis.non_rigid_registrars import OpticalFlowWarper

    reference = f"mouse_apap_{block}_he"
    slide = f"mouse_apap_{block}_cyp2e1"
    he = slides / f"{reference}.ome.tiff"
    cyp = slides / f"{slide}.ome.tiff"

    downsample = shared_downsample([he, cyp], edge)
    grid_um = um_per_px(he) * downsample
    matrix = sidecar_matrix(slides, slide)

    fixed_grey = read_level(he, downsample)
    moving_grey = read_level(cyp, downsample)
    fixed_mask = tissue_mask(fixed_grey)
    fixed = density(fixed_grey, fixed_mask)
    moving = _warp(density(moving_grey, tissue_mask(moving_grey)).astype(float),
                   scaled(matrix, downsample), fixed.shape).astype(np.uint8)
    mask = fixed_mask | (moving > 0)  # union, so a misaligned edge can still move

    start = time.time()
    warped, _grid, dxdy = OpticalFlowWarper().register(
        moving, fixed, mask=mask.astype(np.uint8) * 255)
    seconds = time.time() - start

    agree = agreement(moving, warped, dxdy, mask)
    if agree < MIN_AGREEMENT:
        raise RuntimeError(f"{block}: sampling at p + d agrees with VALIS at only "
                           f"{agree:.3f}, so the stored convention would be wrong")

    shift_um = np.hypot(*dxdy)[mask] * grid_um
    method = (f"valis-{valis.__version__} OpticalFlowWarper (DeepFlow), tissue density "
              f"input at {edge} px, after the sidecar matrix")
    summary = {
        "slide": slide,
        "reference": reference,
        "field": f"{slide}_nonrigid.npz",
        "grid_um_per_px": grid_um,
        "grid_shape_rc": list(fixed.shape),
        "method": method,
        "median_um": round(float(np.median(shift_um)), 1),
        "p95_um": round(float(np.percentile(shift_um, 95)), 1),
        "agreement": round(agree, 4),
        "seconds": round(seconds, 1),
        "registered": datetime.now(UTC).date().isoformat(),
    }

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / summary["field"],
        dx_um=(dxdy[0] * grid_um).astype(np.float32),
        dy_um=(dxdy[1] * grid_um).astype(np.float32),
        grid_um_per_px=grid_um,
        matrix=matrix,  # the field is only valid on top of this matrix
    )
    (out / f"{slide}_nonrigid.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Register every block named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("block", nargs="+", help="dose and animal, e.g. 281mg_m2")
    parser.add_argument("--slides", type=Path, required=True,
                        help="OME-TIFFs with their sidecars")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edge", type=int, default=EDGE_PX, help="longest edge in px")
    args = parser.parse_args()

    for block in args.block:
        s = register_block(block, args.slides, args.out, args.edge)
        print(f"{block:10s} {s['grid_um_per_px']:.2f} µm/px  median {s['median_um']:5.1f} µm"
              f"  p95 {s['p95_um']:5.1f} µm  agreement {s['agreement']:.4f}  {s['seconds']} s")


if __name__ == "__main__":
    main()
