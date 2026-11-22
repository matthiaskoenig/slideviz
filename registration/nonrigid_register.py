"""Refine slide alignment with DeepFlow and save the displacement field and summary.

Two steps in two venvs. `inputs` runs in the slideviz venv: it reads a registered slide
and the reference its sidecar names, in any format slideviz.io.reader knows, and writes
the DeepFlow inputs to <out>/<slide>_input.npz. `flow` runs in this folder's venv, which
holds VALIS: it writes <slide>_nonrigid.npz and <slide>_nonrigid.json, then deletes the inputs.

    uv run python registration/nonrigid_register.py inputs mouse_apap_281mg_m2_cyp2e1 \
        --slides /data/michelle/slides --out /data/michelle/c1/fields
    cd registration && uv run python nonrigid_register.py flow mouse_apap_281mg_m2_cyp2e1 \
        --out /data/michelle/c1/fields
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
    open_cached,
    read_level,
    read_rgb,
    shared_downsample,
    tissue_mask,
)

EDGE_PX = 2048  # longest edge of the DeepFlow grid, ~4 um/px on a mouse slide
MIN_AGREEMENT = 0.99  # own resampling against VALIS's warped image
INPUTS = {"density": "tissue density", "haematoxylin": "haematoxylin channel"}


def code_version() -> str:
    """The repo's commit, marked dirty when files differ from it."""
    import subprocess

    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=Path(__file__).resolve().parent)
    return found.stdout.strip() or "unknown"


def um_per_px(path: Path) -> float:
    """Full-resolution pixel size, as slideviz.io.reader reads it."""
    info, _ = open_cached(path)
    return info.pixel_size_um


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


def haematoxylin(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Haematoxylin density inside the tissue, 1 to 255, zero outside; DAB split off."""
    from skimage.color import rgb2hed

    density = rgb2hed(rgb)[..., 0]
    low, high = np.percentile(density[mask], [1, 99])
    return np.where(mask, np.clip((density - low) / (high - low), 0, 1) * 254 + 1,
                    0).astype(np.uint8)


def inputs(slide: str, slides: Path, out: Path, edge: int,
           source: str = "haematoxylin") -> Path:
    """Write the DeepFlow inputs of a registered slide: its reference, and it on its matrix."""
    from slideviz.data.registration import read_sidecar

    registration = read_sidecar(slides, slide)["registration"]
    reference = registration["reference"]
    fixed_path = slides / read_sidecar(slides, reference)["file"]
    moving_path = slides / read_sidecar(slides, slide)["file"]
    matrix = np.array(registration["matrix"], float)

    downsample = shared_downsample([fixed_path, moving_path], edge)
    grid_um = um_per_px(fixed_path) * downsample

    fixed_grey = read_level(fixed_path, downsample)
    fixed_mask = tissue_mask(fixed_grey)
    if source == "density":
        moving_grey = read_level(moving_path, downsample)
        fixed = density(fixed_grey, fixed_mask)
        moving = _warp(density(moving_grey, tissue_mask(moving_grey)).astype(float),
                       scaled(matrix, downsample), fixed.shape).astype(np.uint8)
    else:
        fixed = haematoxylin(read_rgb(fixed_path, downsample), fixed_mask)
        moving_rgb = read_rgb(moving_path, downsample)
        moving = _warp(haematoxylin(moving_rgb, tissue_mask(moving_rgb.mean(axis=-1)))
                       .astype(float), scaled(matrix, downsample), fixed.shape).astype(np.uint8)
    mask = fixed_mask | (moving > 0)  # union, so a misaligned edge can still move

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{slide}_input.npz"
    np.savez_compressed(path, fixed=fixed, moving=moving, mask=mask, matrix=matrix,
                        grid_um_per_px=grid_um, reference=reference, source=source, edge=edge)
    return path


def flow(slide: str, out: Path) -> dict:
    """Run DeepFlow on one slide's inputs and write its field and summary."""
    import valis
    from valis.non_rigid_registrars import OpticalFlowWarper

    path = out / f"{slide}_input.npz"
    with np.load(path) as stored:
        fixed, moving, mask = stored["fixed"], stored["moving"], stored["mask"]
        matrix, grid_um = stored["matrix"], float(stored["grid_um_per_px"])
        reference, source = str(stored["reference"]), str(stored["source"])
        edge = int(stored["edge"])

    start = time.time()
    warped, _grid, dxdy = OpticalFlowWarper().register(
        moving, fixed, mask=mask.astype(np.uint8) * 255)
    seconds = time.time() - start

    agree = agreement(moving, warped, dxdy, mask)
    if agree < MIN_AGREEMENT:
        raise RuntimeError(f"{slide}: sampling at p + d agrees with VALIS at only "
                           f"{agree:.3f}, so the stored convention would be wrong")

    shift_um = np.hypot(*dxdy)[mask] * grid_um
    method = (f"valis-{valis.__version__} OpticalFlowWarper (DeepFlow), {INPUTS[source]} "
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
        "code": code_version(),
    }

    np.savez_compressed(
        out / summary["field"],
        dx_um=(dxdy[0] * grid_um).astype(np.float32),
        dy_um=(dxdy[1] * grid_um).astype(np.float32),
        grid_um_per_px=grid_um,
        matrix=matrix,  # the field is only valid on top of this matrix
    )
    (out / f"{slide}_nonrigid.json").write_text(json.dumps(summary, indent=2) + "\n")
    path.unlink()
    return summary


def main() -> None:
    """Run one step for every slide named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("inputs", "flow"))
    parser.add_argument("slide", nargs="+",
                        help="registered slide stems, e.g. mouse_apap_281mg_m2_cyp2e1")
    parser.add_argument("--slides", type=Path,
                        help="inputs: slides and sidecars, the registered ones with a matrix")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edge", type=int, default=EDGE_PX, help="longest edge in px")
    parser.add_argument("--input", choices=tuple(INPUTS), default="haematoxylin",
                        help="haematoxylin: the counterstain both slides carry, DAB split off; "
                             "density: inverted grey inside the tissue")
    args = parser.parse_args()
    if args.step == "inputs" and args.slides is None:
        parser.error("inputs needs --slides")

    refused = []
    for slide in args.slide:
        if args.step == "inputs":
            print(f"{slide}: {inputs(slide, args.slides, args.out, args.edge, args.input)}")
            continue
        try:
            s = flow(slide, args.out)
        except RuntimeError as exc:
            print(f"{slide} REFUSED: {exc}")
            refused.append(slide)
            continue
        print(f"{slide} {s['grid_um_per_px']:.2f} µm/px  median {s['median_um']:5.1f} µm"
              f"  p95 {s['p95_um']:5.1f} µm  agreement {s['agreement']:.4f}  {s['seconds']} s")
    if refused:
        raise SystemExit(f"refused, no field written: {', '.join(refused)}")


if __name__ == "__main__":
    main()
