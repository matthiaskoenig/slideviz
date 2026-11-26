"""Register a pair from its outline, then refine on intensity.

A rotation sweep over the tissue outlines fixes the pose to within a degree,
~100 um at the slide edge; maximising gradient correlation refines it from there.
Scale is held at 1.0, the size two sections off one block share.

A pair whose whole outline fails the outline check starts from one piece instead: the
largest moving piece that a reference piece fits at the check's Dice, upright beating
mirrored, sets the starting pose, the refine runs on the whole slide as for every pair,
and the pair registers when that piece pair still overlaps at the check's Dice. Lobes of
different rats fit each other at piece Dice up to 0.95 as well, so an anchored pair's
identity rests on serial_block from the sorting alone.

Runs in the slideviz venv and reads every format slideviz.io.reader knows. Writes
transforms.json, the schema slideviz.data.registration.from_valis_run reads, and
outline_error.csv.

    uv run python registration/outline_register.py <slides_dir> <out_dir> --reference <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from scipy.optimize import minimize
from skimage.filters import sobel
from skimage.measure import label, regionprops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from slideviz.analysis.outline import (
    MIN_DICE,
    SWAP_XY,
    align,
    best_rotation,
    outline_qc,
    warp_mask,
)

POSE_EDGE_PX = 400  # a sweep takes seconds at this size
REFINE_EDGES_PX = (400, 800)

MAX_SHIFT_PX = 60  # the range the outline pose lands in
ANGLE_WINDOW_DEG = 4.0
TISSUE_RIM_PX = 3  # edges kept this far outside the tissue, so its own border counts

ANCHOR_SHARE = 0.25  # pieces down to this share of their slide's largest one can anchor the pose


@cache
def open_cached(path: Path) -> tuple:
    """The slide's (info, zoom levels) from slideviz.io.reader, opened once per process."""
    from slideviz.io.reader import open_slide

    return open_slide(path)


def full_size(path: Path) -> tuple[int, int]:
    """The slide's full-resolution (width, height)."""
    info, _ = open_cached(path)
    return info.width, info.height


def pyramid_level(path: Path, target: float) -> tuple[int, int]:
    """The deepest zoom level at least `target` px wide and its width; 0 is full resolution."""
    _, levels = open_cached(path)
    level = 0
    for candidate, array in enumerate(levels):
        if array.shape[1] < target:
            break
        level = candidate
    return level, levels[level].shape[1]


def read_rgb(path: Path, downsample: float) -> np.ndarray:
    """An RGB view of a slide shrunk by `downsample`, resized with pyvips' Lanczos 3 and gap 2."""
    from PIL import Image

    width, _ = full_size(path)
    target = width / downsample
    level, level_width = pyramid_level(path, target)
    rgb = np.asarray(open_cached(path)[1][level].compute(scheduler="synchronous"))
    if level_width > target:
        scale = target / level_width
        size = (int(level_width * scale + 0.5), int(rgb.shape[0] * scale + 0.5))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS, reducing_gap=2.0))
    return rgb


def read_level(path: Path, downsample: float) -> np.ndarray:
    """A greyscale view of a slide shrunk by `downsample`, its three channels averaged."""
    return read_rgb(path, downsample).mean(axis=-1)


def shared_downsample(paths, edge_px: int) -> float:
    """One shrink factor for every slide in a pair, from the largest of them."""
    longest = max(max(full_size(p)) for p in paths)
    return max(1.0, longest / edge_px)


def on_canvas(image: np.ndarray, shape) -> np.ndarray:
    """The image placed at the origin of a canvas both slides share."""
    canvas = np.zeros(shape, image.dtype)
    rows = min(shape[0], image.shape[0])
    cols = min(shape[1], image.shape[1])
    canvas[:rows, :cols] = image[:rows, :cols]
    return canvas


def tissue_mask(grey: np.ndarray) -> np.ndarray:
    """Tissue, against the synthetic white outside the scan and the glass inside."""
    from skimage.filters import threshold_otsu

    rest = grey < 250  # synthetic white, written outside the scan grid
    if not rest.any():
        raise ValueError("nothing but synthetic white, so the slide did not scan")
    mask = rest & (grey < threshold_otsu(grey[rest]))
    return binary_fill_holes(mask)


def edges(grey: np.ndarray) -> np.ndarray:
    """Edge magnitude scaled to 0..1, the representation shared across stains."""
    inverted = 1.0 - grey.astype(float) / 255.0  # tissue bright, background dark
    found = sobel(inverted)
    spread = found.max() - found.min()
    return (found - found.min()) / spread if spread > 0 else np.zeros_like(found)


def gradient(grey: np.ndarray) -> np.ndarray:
    """Edges in the tissue and a TISSUE_RIM_PX rim around it; the scan border and other background edges are zero."""
    found = edges(grey)
    found[~binary_dilation(tissue_mask(grey), iterations=TISSUE_RIM_PX)] = 0.0
    return found


def _rigid_matrix(angle_deg: float, tx: float, ty: float, centre) -> np.ndarray:
    """An x/y rotation about `centre` followed by a shift, as a 3x3."""
    radians = np.deg2rad(angle_deg)
    cos, sin = np.cos(radians), np.sin(radians)
    rotation = np.array([[cos, -sin], [sin, cos]])
    offset = np.asarray(centre, float) - rotation @ np.asarray(centre, float)
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = offset + [tx, ty]
    return matrix


def _warp(image: np.ndarray, matrix: np.ndarray, shape) -> np.ndarray:
    """Apply an x/y matrix to an image indexed (row, col)."""
    from scipy.ndimage import affine_transform

    row_col = np.linalg.inv(SWAP_XY @ matrix @ SWAP_XY)
    return affine_transform(
        image, row_col[:2, :2], offset=row_col[:2, 2], output_shape=shape, order=1,
    )


def correlation(fixed: np.ndarray, moving: np.ndarray, covered: np.ndarray) -> float:
    """Normalised cross-correlation over the covered pixels."""
    # catches a warp that slid off the canvas
    if covered.sum() < 0.02 * covered.size:
        return 0.0
    a, b = fixed[covered], moving[covered]
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def refine(
    fixed: np.ndarray, moving: np.ndarray, matrix: np.ndarray, window: float,
) -> tuple[np.ndarray, float]:
    """The pose after a Powell search for peak gradient correlation, and that correlation."""
    # one canvas for both, so the transform means the same in each
    shape = (max(fixed.shape[0], moving.shape[0]), max(fixed.shape[1], moving.shape[1]))
    fixed_gradient = on_canvas(gradient(fixed), shape)
    moving_gradient = on_canvas(gradient(moving), shape)
    moving_edges = on_canvas(edges(moving), shape)  # nonzero wherever the moving slide's warp covers the canvas
    centre = np.array(fixed.shape[::-1], float) / 2

    def corrected(parameters):
        """The pose with this (angle, tx, ty) correction applied."""
        angle_deg, tx, ty = parameters
        return _rigid_matrix(angle_deg, tx, ty, centre) @ matrix

    def cost(parameters):
        """One minus the gradient correlation at this correction."""
        angle_deg, tx, ty = parameters
        if abs(angle_deg) > window or max(abs(tx), abs(ty)) > MAX_SHIFT_PX:
            return 1.0  # outside the trust region
        warped = _warp(moving_gradient, corrected(parameters), shape)
        covered = _warp(moving_edges, corrected(parameters), shape) > 0
        return 1.0 - correlation(fixed_gradient, warped, covered)

    result = minimize(cost, np.zeros(3), method="Powell",
                      options={"xtol": 0.01, "ftol": 1e-6, "maxiter": 2000})
    return corrected(result.x), float(1.0 - result.fun)


def pose_from_outline(fixed_mask: np.ndarray, moving_mask: np.ndarray):
    """The outline QC report and the matrix that best overlaps the two masks, in mask pixels."""
    report = outline_qc(fixed_mask, moving_mask)
    matrix, _ = align(fixed_mask, moving_mask, fixed_mask.shape, moving_mask.shape,
                      angle=report.angle)
    return report, matrix


def pieces(mask: np.ndarray) -> list[np.ndarray]:
    """The mask's pieces as masks of its full frame, largest first, down to ANCHOR_SHARE of the largest."""
    labelled = label(mask)
    regions = sorted(regionprops(labelled), key=lambda region: -region.area)
    return [labelled == region.label for region in regions
            if region.area >= ANCHOR_SHARE * regions[0].area]


def cropped(mask: np.ndarray) -> np.ndarray:
    """The mask cut to its bounding box."""
    rows, cols = np.nonzero(mask)
    return mask[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]


def pose_from_piece(fixed_mask: np.ndarray, moving_mask: np.ndarray):
    """The largest moving piece a fixed piece fits at MIN_DICE, upright beating mirrored, as (fixed piece, moving piece, angle, sweep Dice, matrix in mask pixels); None if none fits."""
    fixed_pieces = pieces(fixed_mask)
    for moving_piece in pieces(moving_mask):
        fits = [(best_rotation(cropped(piece), cropped(moving_piece)), piece)
                for piece in fixed_pieces]
        (angle, score, _, mirrored), fixed_piece = max(fits, key=lambda fit: fit[0][1])
        if score >= MIN_DICE and score > mirrored:
            matrix, _ = align(fixed_piece, moving_piece, fixed_mask.shape, moving_mask.shape,
                              angle=angle)
            return fixed_piece, moving_piece, angle, score, matrix
    return None


def mask_dice(fixed_mask: np.ndarray, moving_mask: np.ndarray, matrix) -> float:
    """Dice of the fixed outline and the moving one warped by `matrix`, in the mask's own pixel grid."""
    warped = warp_mask(moving_mask, matrix, fixed_mask.shape)
    return float(2 * (fixed_mask & warped).sum() / (fixed_mask.sum() + warped.sum()))


def overlap_png(fixed: np.ndarray, moving: np.ndarray, matrix, path: Path) -> None:
    """Write fixed gradients in magenta, warped moving ones in green; agreement is grey."""
    from PIL import Image

    shape = (max(fixed.shape[0], moving.shape[0]), max(fixed.shape[1], moving.shape[1]))
    a = on_canvas(gradient(fixed), shape)
    b = _warp(on_canvas(gradient(moving), shape), matrix, shape)
    rgb = np.zeros((*shape, 3), np.uint8)
    rgb[..., 0] = np.clip(a * 500, 0, 255)  # gradients are faint
    rgb[..., 1] = np.clip(b * 500, 0, 255)
    rgb[..., 2] = np.clip(a * 500, 0, 255)
    Image.fromarray(rgb).save(path)


def main() -> None:
    """Register every moving slide in a directory against the reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--reference", required=True, help="the fixed slide's filename")
    args = parser.parse_args()

    from slideviz.io.reader import BACKENDS, suffix_of

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slides = sorted(p for p in args.slide_dir.iterdir() if suffix_of(p) in BACKENDS)
    reference = args.slide_dir / args.reference
    moving_slides = [p for p in slides if p != reference]
    if not reference.exists() or not moving_slides:
        sys.exit(f"need a reference and at least one moving slide in {args.slide_dir}")

    print(f"reference: {reference.name}")
    entries, scores, dices = {}, {}, {}
    for moving_path in moving_slides:
        print(f"\n=== {moving_path.name} ===")

        # one shrink factor for both, so the pose means the same in each
        pose_ds = shared_downsample([reference, moving_path], POSE_EDGE_PX)
        fixed_pose = read_level(reference, pose_ds)
        moving_pose = read_level(moving_path, pose_ds)
        fixed_mask = tissue_mask(fixed_pose)
        moving_mask = tissue_mask(moving_pose)

        report, matrix = pose_from_outline(fixed_mask, moving_mask)
        print(f"  outline: angle {report.angle:.2f} deg, dice {report.dice:.4f}, "
              f"sharpness {report.sharpness:.4f}, {report.verdict}")
        for note in report.notes:
            print(f"    ! {note}")
        anchor = None
        if not report.ok:
            anchor = pose_from_piece(fixed_mask, moving_mask)
            if anchor is None:
                sys.exit("  the pair fails outline QC and no piece fits; not registering it")
            fixed_piece, moving_piece, angle, sweep_dice, matrix = anchor
            pixel_um = open_cached(moving_path)[0].pixel_size_um * pose_ds
            print(f"  anchored on a {moving_piece.sum() * pixel_um**2 / 1e6:.1f} mm2 piece: "
                  f"angle {angle:.2f} deg, piece dice {sweep_dice:.4f}")

        # coarse to fine, each level improving on the one before
        for edge_px in REFINE_EDGES_PX:
            level_ds = shared_downsample([reference, moving_path], edge_px)
            rescale = pose_ds / level_ds
            up = np.diag([rescale, rescale, 1.0])
            matrix = up @ matrix @ np.linalg.inv(up)

            matrix, score = refine(
                read_level(reference, level_ds), read_level(moving_path, level_ds),
                matrix, ANGLE_WINDOW_DEG,
            )
            rotation = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
            print(f"  refined at 1/{level_ds:.0f}: angle {rotation:.3f} deg, "
                  f"correlation {score:.4f}")
            matrix = np.linalg.inv(up) @ matrix @ up
        last_ds = shared_downsample([reference, moving_path], REFINE_EDGES_PX[-1])
        overlap_png(read_level(reference, last_ds), read_level(moving_path, last_ds),
                    np.diag([pose_ds / last_ds] * 2 + [1.0]) @ matrix
                    @ np.linalg.inv(np.diag([pose_ds / last_ds] * 2 + [1.0])),
                    args.out_dir / f"{moving_path.name.split('.')[0]}_overlap.png")

        # the pose is in shrunk pixels, the stored matrix in full resolution
        fixed_width, fixed_height = full_size(reference)
        moving_width, moving_height = full_size(moving_path)
        to_full = np.diag([fixed_width / fixed_pose.shape[1],
                           fixed_height / fixed_pose.shape[0], 1.0])
        from_full = np.diag([moving_pose.shape[1] / moving_width,
                             moving_pose.shape[0] / moving_height, 1.0])
        full = to_full @ matrix @ from_full

        name = moving_path.name.split(".")[0]
        final_dice = mask_dice(fixed_mask, moving_mask, matrix)
        print(f"  final outline dice {final_dice:.4f}")
        dices[name] = final_dice
        entries[name] = {
            "source": str(moving_path),
            "matrix": full.tolist(),
            # a rotation and a shift, so exactly affine
            "residual_px": 0.0,
            "slide_shape_rc": [moving_height, moving_width],
            "outline_dice": final_dice,
        }
        if anchor is not None:
            anchor_dice = mask_dice(fixed_piece, moving_piece, matrix)
            print(f"  final anchor piece dice {anchor_dice:.4f}")
            if anchor_dice < MIN_DICE:
                sys.exit(f"  the refine leaves the anchor pieces at dice {anchor_dice:.4f}, "
                         f"below {MIN_DICE:.2f}; not registering it")
            entries[name]["anchor_dice"] = anchor_dice
        scores[name] = score
        print(f"  scale {float(np.sqrt(abs(np.linalg.det(full[:2, :2])))):.4f}")

    (args.out_dir / "transforms.json").write_text(json.dumps({
        "reference": str(reference),
        "slides": entries,
    }, indent=2) + "\n")
    print(f"\nwrote transforms.json ({len(entries)} slides)")

    # per moving slide, the gradient correlation after the refine
    with (args.out_dir / "outline_error.csv").open("w") as handle:
        handle.write("name,correlation\n")
        for name, score in scores.items():
            handle.write(f"{name},{score:.6f}\n")


if __name__ == "__main__":
    main()
