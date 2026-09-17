"""Convert predicted tiles into smoothed CVAT polygons for manual review.

These model-generated suggestions must be reviewed, corrected, and recorded as
model-assisted annotations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import closing, disk, remove_small_holes

log = logging.getLogger(__name__)

# below this many tiles a region is noise, not a lesion worth drawing
MIN_TILES = 4

# how far a traced outline may sit from the mask, in tile widths; keeps point counts sane
TOLERANCE_TILES = 0.4


def flagged_mask(rows: list[dict], scores: np.ndarray, threshold: float) -> np.ndarray:
    """The tile grid as a boolean mask of what the model flagged."""
    height = max(r["row"] for r in rows) + 1
    width = max(r["col"] for r in rows) + 1
    mask = np.zeros((height, width), dtype=bool)
    for row, score in zip(rows, scores, strict=True):
        mask[row["row"], row["col"]] = score >= threshold
    return mask


def clean_mask(mask: np.ndarray, min_tiles: int = MIN_TILES) -> np.ndarray:
    """Close one-tile gaps and drop specks, so tracing gives regions not confetti."""
    closed = closing(mask, disk(1))
    # max_size removes holes of that size or smaller, so one less keeps min_tiles intact
    return remove_small_holes(closed, max_size=min_tiles - 1)


def mask_to_polygons(
    mask: np.ndarray, min_tiles: int = MIN_TILES, tolerance: float = TOLERANCE_TILES
) -> list[np.ndarray]:
    """Outlines of every flagged region, in tile coordinates."""
    polygons = []
    for contour in find_contours(mask.astype(float), 0.5):
        simplified = approximate_polygon(contour, tolerance=tolerance)
        if len(simplified) < 4:
            continue  # fewer than four points is not a region
        # shoelace area in tile units, so the threshold means "this many tiles"
        y, x = simplified[:, 0], simplified[:, 1]
        area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
        if area >= min_tiles:
            polygons.append(simplified)
    return polygons


def to_overview(
    polygon: np.ndarray, size_px: int, level_um_per_px: float, overview_um_per_px: float
) -> list[float]:
    """One tile-space polygon as the flat x,y list CVAT stores, in overview pixels."""
    # tile index to level pixels, then level pixels to overview pixels
    scale = size_px * level_um_per_px / overview_um_per_px
    points = polygon * scale
    return [float(v) for point in points for v in (point[1], point[0])]  # CVAT wants x,y


def suggest_polygons(
    predictions: Path,
    overview: dict,
    level_um_per_px: float,
    threshold: float = 0.5,
    min_tiles: int = MIN_TILES,
) -> dict:
    """One slide's predicted tiles as CVAT-shaped polygons, with their provenance."""
    record = json.loads(predictions.read_text())
    rows = record["tiles"]
    scores = np.array([t["predicted"] for t in rows], dtype=float)

    mask = clean_mask(flagged_mask(rows, scores, threshold), min_tiles)
    polygons = mask_to_polygons(mask, min_tiles)
    size_px = record["size_px"]  # the grid's tile size, recorded once per slide

    log.info(
        "%s: %d tiles flagged at %.2f, %d polygons",
        record["animal"],
        int(mask.sum()),
        threshold,
        len(polygons),
    )
    return {
        "slide": record.get("slide"),
        "animal": record["animal"],
        "frame": overview.get("frame"),
        "width": overview["width"],
        "height": overview["height"],
        "model_generated": True,  # review and correct before these count as annotation
        "encoder": record.get("encoder"),
        "trained_on": record.get("trained_on"),
        "threshold": threshold,
        "min_tiles": min_tiles,
        "n_tiles_flagged": int(mask.sum()),
        "polygons": [
            {
                "label": "necrosis",
                "points": to_overview(
                    p, size_px, level_um_per_px, overview["um_per_px"]
                ),
            }
            for p in polygons
        ],
    }
