"""Align sections by their outlines, and judge a pair before registering it.

Centroids give the translation, a rotation sweep over binary masks gives the angle,
scored by Dice. Uses shape alone, for stain pairs that differ in local texture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, rotate, shift, zoom
from skimage.measure import label, regionprops

# matrices are built (x, y) while masks are indexed (row, col)
SWAP_XY = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

SWEEP_STEP_DEG = 2.0
REFINE_STEP_DEG = 0.25  # ~25 um at the slide edge
REFINE_WINDOW_DEG = 3.0

# beyond this from the best angle counts as a rival peak
PEAK_SEPARATION_DEG = 20.0

MIN_DICE = 0.90  # good blocks span 0.9235 to 0.9900
WEAK_SHARPNESS = 0.03  # below this, confirm the angle by intensity
AREA_RATIO_LIMIT = 0.25

MASK_EDGE_PX = 400
PROCESSED_THRESHOLD = 8


def principal_axis(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Angle of the largest blob's long axis in degrees, and its centroid (row, col)."""
    biggest = max(regionprops(label(mask)), key=lambda region: region.area)
    return float(np.rad2deg(biggest.orientation)), np.array(biggest.centroid)


def elongation(mask: np.ndarray) -> float:
    """Ratio of the mask's principal axes."""
    biggest = max(regionprops(label(mask)), key=lambda region: region.area)
    if biggest.axis_minor_length == 0:
        return float("inf")
    return float(biggest.axis_major_length / biggest.axis_minor_length)


def dice(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Overlap of two masks after matching their centroids, on a shared canvas."""
    height = max(fixed.shape[0], moving.shape[0])
    width = max(fixed.shape[1], moving.shape[1])

    a = np.zeros((height, width), bool)
    b = np.zeros((height, width), bool)
    a[: fixed.shape[0], : fixed.shape[1]] = fixed
    b[: moving.shape[0], : moving.shape[1]] = moving
    if not a.any() or not b.any():
        return 0.0

    offset = np.argwhere(a).mean(axis=0) - np.argwhere(b).mean(axis=0)
    b = shift(b.astype(float), offset, order=0) > 0.5
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def warp_mask(mask: np.ndarray, matrix: np.ndarray, shape) -> np.ndarray:
    """Apply an x/y matrix to a mask, in the same pixel grid it was built in."""
    from scipy.ndimage import affine_transform

    row_col = SWAP_XY @ matrix @ SWAP_XY
    inverse = np.linalg.inv(row_col)
    warped = affine_transform(
        mask.astype(float), inverse[:2, :2], offset=inverse[:2, 2],
        output_shape=shape, order=0,
    )
    return warped > 0.5


def _turn(mask: np.ndarray, angle: float) -> np.ndarray:
    """The mask rotated by `angle`, in scipy's sense and its enlarged frame."""
    return rotate(mask.astype(float), angle, reshape=True, order=1) > 0.5


def sweep_rotation(
    fixed: np.ndarray, moving: np.ndarray, step: float = SWEEP_STEP_DEG,
    mirrored: bool = False,
) -> list[tuple[float, float]]:
    """Dice at every angle on a grid, as (angle, dice), highest first."""
    if mirrored:
        moving = moving[:, ::-1]

    scored = [
        (float(angle), dice(fixed, _turn(moving, float(angle))))
        for angle in np.arange(0.0, 360.0, step)
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def refine_rotation(
    fixed: np.ndarray, moving: np.ndarray, coarse: float,
    step: float = REFINE_STEP_DEG, window: float = REFINE_WINDOW_DEG,
) -> tuple[float, float]:
    """The best angle near `coarse`, and its Dice, on a finer grid."""
    angles = np.arange(coarse - window, coarse + window + step / 2, step)
    scored = [(float(a), dice(fixed, _turn(moving, float(a)))) for a in angles]
    return max(scored, key=lambda pair: pair[1])


def peak_sharpness(scored: list[tuple[float, float]]) -> float:
    """How far the best angle beats the best one more than PEAK_SEPARATION_DEG away."""
    if not scored:
        return 0.0
    best_angle, best_score = scored[0]
    rivals = [
        score for angle, score in scored
        if abs((angle - best_angle + 180.0) % 360.0 - 180.0) > PEAK_SEPARATION_DEG
    ]
    return float(best_score - max(rivals)) if rivals else float(best_score)


def best_rotation(
    fixed: np.ndarray, moving: np.ndarray, step: float = SWEEP_STEP_DEG,
    refine: bool = True,
) -> tuple[float, float, float, float]:
    """Angle, its Dice, the peak's sharpness, and the best Dice when mirrored.

    Mirrored beating upright means a section was mounted face down.
    """
    scored = sweep_rotation(fixed, moving, step)
    angle, score = scored[0]
    sharpness = peak_sharpness(scored)

    if refine:
        angle, score = refine_rotation(fixed, moving, angle)

    mirrored = sweep_rotation(fixed, moving, step, mirrored=True)[0][1]
    return angle, score, sharpness, mirrored


def align(
    fixed: np.ndarray, moving: np.ndarray, fixed_shape_rc, moving_shape_rc,
    angle: float | None = None,
) -> tuple[np.ndarray, float]:
    """A full-resolution x/y matrix mapping the moving slide onto the fixed one.

    Masks are at some pyramid level; the matrix is rescaled to full resolution.
    `angle` builds the matrix for an angle found elsewhere.
    """
    if angle is None:
        angle, *_ = best_rotation(fixed, moving)

    # rotate the grid through the same call that warps the mask: rotate(reshape=True)
    # works in (row, col) and recentres, a transpose away from a hand-composed one
    turned = _turn(moving, angle)
    rows, cols = moving.shape
    grid_rc = np.array(
        [[r, c] for r in np.linspace(0, rows - 1, 5) for c in np.linspace(0, cols - 1, 5)],
        float,
    )

    radians = np.deg2rad(angle)
    cos, sin = np.cos(radians), np.sin(radians)
    # measured off rotate() with a probe grid, to 0.5 px
    rotation = np.array([[cos, -sin], [sin, cos]])
    moving_centre = (np.array(moving.shape, float) - 1) / 2
    turned_centre = (np.array(turned.shape, float) - 1) / 2
    rotated_rc = (grid_rc - moving_centre) @ rotation.T + turned_centre

    # the centroid shift `dice` applies
    offset = np.argwhere(fixed).mean(axis=0) - np.argwhere(turned).mean(axis=0)
    warped_rc = rotated_rc + offset

    # fit in (x, y), the stored matrix's convention
    grid = grid_rc[:, ::-1]
    warped = warped_rc[:, ::-1]
    padded = np.hstack([grid, np.ones((len(grid), 1))])
    fit, *_ = np.linalg.lstsq(padded, warped, rcond=None)
    in_mask = np.vstack([fit.T, [0, 0, 1]])

    # score the matrix itself
    achieved = dice(fixed, warp_mask(moving, in_mask, fixed.shape))

    # mask pixels -> full resolution, each side by its own factor
    to_full = np.diag([fixed_shape_rc[1] / fixed.shape[1],
                       fixed_shape_rc[0] / fixed.shape[0], 1.0])
    from_full = np.diag([moving.shape[1] / moving_shape_rc[1],
                         moving.shape[0] / moving_shape_rc[0], 1.0])
    return to_full @ in_mask @ from_full, achieved


@dataclass
class OutlineQC:
    """What the outlines say about a pair."""

    angle: float  # degrees, in scipy.ndimage.rotate's sense
    dice: float
    sharpness: float
    mirrored_dice: float
    area_ratio: float  # moving over fixed
    elongation: float  # of the fixed mask
    verdict: str
    notes: list[str]

    @property
    def ok(self) -> bool:
        """Whether the pair is worth registering."""
        return self.verdict != "reject"


def outline_qc(fixed: np.ndarray, moving: np.ndarray) -> OutlineQC:
    """Judge a pair by its outlines alone.

    Rejects a mismatched pair (low Dice at every angle), a face-down section
    (mirrored beats upright) and a large area difference. Flags a round outline
    (low sharpness) as weak.
    """
    angle, score, sharpness, mirrored = best_rotation(fixed, moving)
    ratio = float(moving.sum() / fixed.sum()) if fixed.sum() else 0.0

    notes: list[str] = []
    verdict = "pass"

    if score < MIN_DICE:
        notes.append(f"dice {score:.4f} is below {MIN_DICE:.2f}, so these may not be a pair")
        verdict = "reject"

    if mirrored > score:
        notes.append(
            f"mirrored dice {mirrored:.4f} beats upright {score:.4f}, "
            "so a section may be mounted face down"
        )
        verdict = "reject"

    if abs(ratio - 1.0) > AREA_RATIO_LIMIT:
        notes.append(f"areas differ by {abs(ratio - 1.0):.0%}, more than two sections should")
        verdict = "reject"

    if sharpness < WEAK_SHARPNESS and verdict == "pass":
        notes.append(
            f"sharpness {sharpness:.4f} is low, so the outline does not fix the "
            "rotation on its own; confirm the angle by intensity"
        )
        verdict = "weak"

    return OutlineQC(
        angle=angle, dice=score, sharpness=sharpness, mirrored_dice=mirrored,
        area_ratio=ratio, elongation=elongation(fixed), verdict=verdict, notes=notes,
    )


def mask_from_png(path, size: int = MASK_EDGE_PX) -> np.ndarray:
    """A filled tissue mask from one of VALIS's processed images."""
    from PIL import Image

    grey = np.asarray(Image.open(path).convert("L"))
    mask = binary_fill_holes(grey > PROCESSED_THRESHOLD)
    scaled = zoom(mask.astype(float), size / max(mask.shape), order=1) > 0.5
    return binary_fill_holes(scaled)


def masks_from_valis_run(run_dir, block: str) -> tuple[np.ndarray, np.ndarray]:
    """The (H&E, CYP2E1) masks for a block, from its VALIS run directory."""
    from pathlib import Path

    processed = Path(run_dir) / "slides" / "processed"
    fixed = processed / f"mouse_apap_{block}_he.png"
    moving = processed / f"mouse_apap_{block}_cyp2e1.png"
    for path in (fixed, moving):
        if not path.exists():
            raise FileNotFoundError(f"no processed image at {path}")
    return mask_from_png(fixed), mask_from_png(moving)
