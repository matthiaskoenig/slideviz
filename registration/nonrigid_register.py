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
from outline_register import (
    _warp,
    full_size,
    read_level,
    shared_downsample,
    tissue_mask,
)

EDGE_PX = 2048  # RAFT tore the tissue at this size, DeepFlow matched its 1024 px result
MIN_AGREEMENT = 0.99  # own resampling against VALIS's warped image
# good blocks move 7–71 um; edge DAB reached 307 um
MAX_MEDIAN_UM = 100.0


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


def read_rgb(path: Path, downsample: float) -> np.ndarray:
    """An RGB view of a slide at the shared shrink, from its three one-band pages."""
    import pyvips

    width, _ = full_size(path)
    target = width / downsample
    level = -1  # deepest SubIFD still at or above target, so the last step shrinks
    for candidate in range(16):
        try:
            probe = pyvips.Image.new_from_file(str(path), page=0, subifd=candidate)
        except pyvips.Error:
            break
        if probe.width < target:
            break
        level = candidate

    bands = []
    for page in range(3):
        image = pyvips.Image.new_from_file(str(path), page=page, subifd=level)
        image = image.resize(target / image.width)
        bands.append(np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8,
                                shape=[image.height, image.width, image.bands])[..., 0])
    rows = min(b.shape[0] for b in bands)
    cols = min(b.shape[1] for b in bands)
    return np.dstack([b[:rows, :cols] for b in bands])


def haematoxylin(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Haematoxylin density inside the tissue, 1 to 255, zero outside; DAB split off."""
    from skimage.color import rgb2hed

    density = rgb2hed(rgb)[..., 0]
    low, high = np.percentile(density[mask], [1, 99])
    return np.where(mask, np.clip((density - low) / (high - low), 0, 1) * 254 + 1,
                    0).astype(np.uint8)


def register_block(block: str, slides: Path, out: Path, edge: int,
                   source: str = "density") -> dict:
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
    if source == "haematoxylin":
        fixed = haematoxylin(read_rgb(he, downsample), fixed_mask)
        moving_rgb = read_rgb(cyp, downsample)
        moving = _warp(haematoxylin(moving_rgb, tissue_mask(moving_rgb.mean(axis=-1)))
                       .astype(float), scaled(matrix, downsample), fixed.shape).astype(np.uint8)
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
    if np.median(shift_um) > MAX_MEDIAN_UM:
        raise RuntimeError(f"{block}: median shift {np.median(shift_um):.0f} µm is above "
                           f"{MAX_MEDIAN_UM:.0f} µm, so the field is matching stain, not tissue")
    inputs = {"density": "tissue density", "haematoxylin": "haematoxylin channel"}
    method = (f"valis-{valis.__version__} OpticalFlowWarper (DeepFlow), {inputs[source]} "
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
    parser.add_argument("--input", choices=("density", "haematoxylin"), default="density",
                        help="haematoxylin leaves DAB out, for a slide whose DAB pattern the "
                             "flow would otherwise follow")
    args = parser.parse_args()

    refused = []
    for block in args.block:
        try:
            s = register_block(block, args.slides, args.out, args.edge, args.input)
        except RuntimeError as exc:
            print(f"{block:10s} REFUSED: {exc}")
            refused.append(block)
            continue
        print(f"{block:10s} {s['grid_um_per_px']:.2f} µm/px  median {s['median_um']:5.1f} µm"
              f"  p95 {s['p95_um']:5.1f} µm  agreement {s['agreement']:.4f}  {s['seconds']} s")
    if refused:
        raise SystemExit(f"refused, no field written: {', '.join(refused)}")


if __name__ == "__main__":
    main()
