"""One PNG with every registered pair drawn as two tissue outlines.

    uv run python scripts/registration_contact_sheet.py \
        --slides <APAP_tiff> --transforms <transforms> --out <contact_sheet.png>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import affine_transform, binary_dilation

from slideviz.analysis.tissue import tissue_mask
from slideviz.io.reader import open_slide

EDGE_PX = 600
SWAP_XY = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
COLUMNS = 5
PANEL_PX = 380
CAPTION_PX = 26
LINE_PX = 2


def outline(mask: np.ndarray) -> np.ndarray:
    """The mask's border as a line."""
    return binary_dilation(mask, iterations=LINE_PX) & ~mask


def warp_mask(mask: np.ndarray, matrix: np.ndarray, downsample: float, shape) -> np.ndarray:
    """A mask under a level-0 matrix, applied in the mask's own coarser grid."""
    scaled = matrix.copy()
    scaled[:2, 2] /= downsample
    row_col = SWAP_XY @ scaled @ SWAP_XY
    inverse = np.linalg.inv(row_col)
    warped = affine_transform(
        mask.astype(float), inverse[:2, :2], offset=inverse[:2, 2], output_shape=shape, order=0
    )
    return warped > 0.5


def decompose(m: np.ndarray) -> tuple[float, float]:
    """Mean scale and rotation in degrees, for the panel caption."""
    sx = math.hypot(m[0][0], m[1][0])
    sy = math.hypot(m[0][1], m[1][1])
    return (sx + sy) / 2, math.degrees(math.atan2(m[1][0], m[0][0]))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two masks already in the same grid."""
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 0.0


def pair_outlines(slides: Path, block: str, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """H&E and registered CYP2E1 outlines on one canvas, with their mask Dice."""
    fixed, _, fixed_um = tissue_mask(
        slides / f"mouse_apap_{block}_he.ome.tiff", target_edge_px=EDGE_PX
    )
    moving, _, moving_um = tissue_mask(
        slides / f"mouse_apap_{block}_cyp2e1.ome.tiff", target_edge_px=EDGE_PX
    )

    if abs(moving_um - fixed_um) / fixed_um > 0.02:
        zoom = moving_um / fixed_um
        grid = np.array([[zoom, 0.0, 0.0], [0.0, zoom, 0.0], [0.0, 0.0, 1.0]])
        moving = affine_transform(
            moving.astype(float), np.linalg.inv(grid)[:2, :2],
            output_shape=(int(moving.shape[0] / zoom), int(moving.shape[1] / zoom)), order=0,
        ) > 0.5

    info, _ = open_slide(slides / f"mouse_apap_{block}_he.ome.tiff")
    downsample = fixed_um / info.pixel_size_um
    warped = warp_mask(moving, matrix, downsample, fixed.shape)
    rgb = np.zeros((*fixed.shape, 3), dtype=np.uint8)
    rgb[outline(fixed)] = (0, 255, 0)
    rgb[outline(warped)] = (255, 0, 255)
    return rgb, dice(fixed, warped)


def read_transforms(transforms: Path) -> dict[str, np.ndarray]:
    """The CYP2E1 matrix per block, without superseded fits."""
    found = {}
    for path in sorted(transforms.glob("*.json")):
        if "." in path.stem or path.stem.endswith(("_euclid", "_reflect")):
            continue
        record = json.loads(path.read_text())
        reference = Path(record["reference"]).name.split(".")[0]
        for name, entry in record["slides"].items():
            if name != reference:
                found[path.stem] = np.array(entry["matrix"], float)
    return found


def panel(image: np.ndarray, caption: str) -> Image.Image:
    """One square panel with its caption underneath."""
    thumbnail = Image.fromarray(image)
    thumbnail.thumbnail((PANEL_PX, PANEL_PX), Image.NEAREST)
    tile = Image.new("RGB", (PANEL_PX, PANEL_PX + CAPTION_PX), "black")
    tile.paste(thumbnail, ((PANEL_PX - thumbnail.width) // 2, (PANEL_PX - thumbnail.height) // 2))
    ImageDraw.Draw(tile).text((4, PANEL_PX + 4), caption, fill="white")
    return tile


def main() -> None:
    """Build the sheet and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    matrices = read_transforms(args.transforms)
    print(f"{len(matrices)} blocks")

    panels = []
    for block, matrix in sorted(matrices.items()):
        scale, rotation = decompose(matrix)
        try:
            image, overlap = pair_outlines(args.slides, block, matrix)
        except (FileNotFoundError, ValueError) as error:
            print(f"  {block}: {error}")
            panels.append(panel(np.zeros((8, 8, 3), np.uint8), f"{block}  FAILED"))
            continue
        print(f"  {block}: dice {overlap:.3f}  scale {scale:.3f}  rot {rotation:+.1f}")
        panels.append(panel(image, f"{block}  dice {overlap:.3f}  r {rotation:+.1f}"))

    rows = math.ceil(len(panels) / COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * PANEL_PX, rows * (PANEL_PX + CAPTION_PX)), "black")
    for i, tile in enumerate(panels):
        sheet.paste(tile, ((i % COLUMNS) * PANEL_PX, (i // COLUMNS) * (PANEL_PX + CAPTION_PX)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  {sheet.width}x{sheet.height}")


if __name__ == "__main__":
    main()
