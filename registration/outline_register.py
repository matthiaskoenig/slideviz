"""Register a pair from its outline, then refine on intensity.

For blocks where VALIS's feature matching fails on the H&E to DAB stain change,
fitting a transform to a handful of keypoints and returning scales like 0.653 and
1.586 alongside a plausible error.

A rotation sweep over the tissue outlines fixes the pose to within a degree,
~100 um at the slide edge; maximising gradient correlation refines it from there.
Scale is held at 1.0, the size two sections off one block share.

Runs in this folder's venv. Writes transforms.json in valis_register.py's schema.

    uv run python outline_register.py <slides_dir> <out_dir> --reference <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, zoom
from scipy.optimize import minimize
from skimage.filters import sobel

POSE_EDGE_PX = 400  # a sweep takes seconds at this size
REFINE_EDGES_PX = (400, 800)

MAX_SHIFT_PX = 60  # the range the outline pose lands in
ANGLE_WINDOW_DEG = 4.0


def full_size(path: Path) -> tuple[int, int]:
    """The slide's full-resolution (width, height)."""
    import pyvips

    image = pyvips.Image.new_from_file(str(path), page=0)
    return image.width, image.height


def read_level(path: Path, downsample: float) -> np.ndarray:
    """A greyscale view of a slide, shrunk by a factor shared across the pair.

    Shared so both land in one pixel grid. bioformats2raw writes brightfield as
    three one-band pages, averaged here. Reads from the SubIFD pyramid.
    """
    import pyvips

    width, _ = full_size(path)
    target = width / downsample

    # deepest level still at or above target, so the last step shrinks
    level, level_width = -1, width
    for candidate in range(16):
        try:
            probe = pyvips.Image.new_from_file(str(path), page=0, subifd=candidate)
        except pyvips.Error:
            break
        if probe.width < target:
            break
        level, level_width = candidate, probe.width

    channels = []
    for page in range(3):
        try:
            image = pyvips.Image.new_from_file(str(path), page=page, subifd=level)
        except pyvips.Error:
            break
        if image.width != level_width:  # a thumbnail page, so the channels ended
            break
        if image.width > target:
            image = image.resize(target / image.width)
        data = np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8,
                          shape=[image.height, image.width, image.bands])
        channels.append(data[..., :3].mean(axis=-1) if data.shape[-1] >= 3
                        else data[..., 0])

    smallest = min(c.shape for c in channels)
    return np.mean([c[:smallest[0], :smallest[1]] for c in channels], axis=0)


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


def gradient(grey: np.ndarray) -> np.ndarray:
    """Edge magnitude, the representation shared across stains."""
    inverted = 1.0 - grey.astype(float) / 255.0  # tissue bright, background dark
    edges = sobel(inverted)
    spread = edges.max() - edges.min()
    return (edges - edges.min()) / spread if spread > 0 else np.zeros_like(edges)


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

    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    row_col = np.linalg.inv(swap @ matrix @ swap)
    return affine_transform(
        image, row_col[:2, :2], offset=row_col[:2, 2], output_shape=shape, order=1,
    )


def correlation(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Normalised cross-correlation over the pixels the warp actually covers."""
    covered = moving > 0
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
    """Nudge a pose until gradient correlation peaks.

    Searches a correction on top of the pose, starting at identity. Powell, since
    resampling is discontinuous.
    """
    # one canvas for both, so the transform means the same in each
    shape = (max(fixed.shape[0], moving.shape[0]), max(fixed.shape[1], moving.shape[1]))
    fixed_gradient = on_canvas(gradient(fixed), shape)
    moving_gradient = on_canvas(gradient(moving), shape)
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
        return 1.0 - correlation(fixed_gradient, warped)

    result = minimize(cost, np.zeros(3), method="Powell",
                      options={"xtol": 0.01, "ftol": 1e-6, "maxiter": 2000})
    return corrected(result.x), float(1.0 - result.fun)


def pose_from_outline(fixed_mask: np.ndarray, moving_mask: np.ndarray):
    """The matrix that best overlaps two outlines, in mask pixels, and the QC report.

    Returns align's matrix whole: its rotation and shift are meaningful together.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from slideviz.analysis.outline import align, outline_qc

    report = outline_qc(fixed_mask, moving_mask)
    matrix, _ = align(fixed_mask, moving_mask, fixed_mask.shape, moving_mask.shape,
                      angle=report.angle)
    return report, matrix


def overlap_png(fixed: np.ndarray, moving: np.ndarray, matrix, path: Path) -> None:
    """Fixed slide in magenta, warped moving one in green; agreement is grey.

    The same view VALIS writes.
    """
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
    parser.add_argument("--angle", type=float, default=None,
                        help="skip the sweep and refine from this angle instead")
    parser.add_argument("--no-refine", action="store_true",
                        help="stop at the outline pose, for comparison")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slides = sorted(p for p in args.slide_dir.iterdir()
                    if p.name.endswith((".ome.tiff", ".tiff", ".tif", ".ndpi")))
    reference = args.slide_dir / args.reference
    moving_slides = [p for p in slides if p != reference]
    if not reference.exists() or not moving_slides:
        sys.exit(f"need a reference and at least one moving slide in {args.slide_dir}")

    print(f"reference: {reference.name}")
    entries, scores = {}, {}
    for moving_path in moving_slides:
        print(f"\n=== {moving_path.name} ===")

        # one shrink factor for both, so the pose means the same in each
        pose_ds = shared_downsample([reference, moving_path], POSE_EDGE_PX)
        fixed_pose = read_level(reference, pose_ds)
        moving_pose = read_level(moving_path, pose_ds)
        fixed_mask = binary_fill_holes(tissue_mask(fixed_pose))
        moving_mask = binary_fill_holes(tissue_mask(moving_pose))

        report, matrix = pose_from_outline(fixed_mask, moving_mask)
        if args.angle is not None:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
            from slideviz.analysis.outline import align

            matrix, _ = align(fixed_mask, moving_mask, fixed_mask.shape,
                              moving_mask.shape, angle=args.angle)

        print(f"  outline: angle {report.angle:.2f} deg, dice {report.dice:.4f}, "
              f"sharpness {report.sharpness:.4f}, {report.verdict}")
        for note in report.notes:
            print(f"    ! {note}")
        if report.verdict == "reject":
            sys.exit("  the pair fails outline QC; not registering it")

        score = report.dice
        if not args.no_refine:
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
        import pyvips

        fixed_image = pyvips.Image.new_from_file(str(reference), n=1)
        moving_image = pyvips.Image.new_from_file(str(moving_path), n=1)
        to_full = np.diag([fixed_image.width / fixed_pose.shape[1],
                           fixed_image.height / fixed_pose.shape[0], 1.0])
        from_full = np.diag([moving_pose.shape[1] / moving_image.width,
                             moving_pose.shape[0] / moving_image.height, 1.0])
        full = to_full @ matrix @ from_full

        name = moving_path.name.split(".")[0]
        entries[name] = {
            "source": str(moving_path),
            "matrix": full.tolist(),
            # a rotation and a shift, so exactly affine
            "residual_px": 0.0,
            "slide_shape_rc": [moving_image.height, moving_image.width],
        }
        scores[name] = score
        print(f"  scale {float(np.sqrt(abs(np.linalg.det(full[:2, :2])))):.4f}")

    (args.out_dir / "transforms.json").write_text(json.dumps({
        "reference": str(reference),
        "slides": entries,
    }, indent=2) + "\n")
    print(f"\nwrote transforms.json ({len(entries)} slides)")

    # a correlation, under its own name: rigid_D measures matched keypoints
    with (args.out_dir / "outline_error.csv").open("w") as handle:
        handle.write("name,correlation\n")
        for name, score in scores.items():
            handle.write(f"{name},{score:.6f}\n")


if __name__ == "__main__":
    main()
