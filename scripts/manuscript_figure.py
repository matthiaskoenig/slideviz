"""Manuscript figure: per dose, whole H&E, zooms with necrosis and CYP2E1 segmentation, model rows, necrotic area.

    uv run --with matplotlib python scripts/manuscript_figure.py --slides <APAP_tiff> \
        --field-scores <predict_field.py output> --results <necrosis_results_nested_area.json> \
        --out <figures>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates

from slideviz.analysis.dab import glass_colour, relative_readout
from slideviz.analysis.tissue import mask_from_level
from slideviz.analysis.tissue import pick_level as outline_level
from slideviz.analysis.warp import load_field, nonrigid_levels
from slideviz.data.schema import Registration
from slideviz.io.ome_tiff import read_pyramid

# representative animal per dose, and its zoom centre in H&E level-0 px (row, col)
BLOCKS = {
    0: ("000mg_m1", (14790, 21250)),
    89: ("089mg_m1", (17170, 25670)),
    158: ("158mg_m2", (16965, 30042)),
    281: ("281mg_m1", (4250, 12750)),
    375: ("375mg_m3", (2890, 19210)),
    500: ("500mg_m1", (4250, 19890)),
}

ZOOM_UM = 1000.0
ZOOM_UM_PER_PX = 0.9
READOUT_UM_PER_PX = 1.8  # the CYP2E1 readout's working resolution
READOUT_MARGIN_UM = 600.0  # more than one readout window, so the field matches the whole-slide run
THUMB_EDGE_PX = 1200
THUMB_PAD = 1.04  # each slide fills its square up to this thin white border

ROWS = ("H&E", "H&E", "Necrosis\nsegmentation", "CYP2E1", "CYP2E1\nsegmentation",
        "Model\nCYP2E1", "Model\nnecrosis")
# negative or surviving, positive or necrotic, outside tissue
NECROSIS_COLOURS = ListedColormap(["#ececf4", "#c0392b", "white"])
CYP2E1_COLOURS = ListedColormap(["#ececf4", "#8b4513", "white"])
# drawn on the stain; CYP2E1 brown on the brown stain would vanish, so it is blue there
NECROSIS_OVERLAY = "#c0392b"
CYP2E1_OVERLAY = "#1f5fbf"
OVERLAY_ALPHA = 0.4
STYLES = ("panels", "overlay")

# page layout in mm; the panel size follows from the height, the width from the panels
MM = 1 / 25.4
HEIGHT_MM = 245
LABEL_MM = 14.0  # row labels left of the grid
GAP_MM = 1.0
TITLE_MM = 6.0
PLOT_GAP_MM = 11.0
PLOT_MM = 38.0
BOTTOM_MM = 10.0
RIGHT_MM = 1.0


def pick_level(levels, level0_um: float, target_um: float) -> int:
    """The coarsest level at or finer than `target_um`."""
    chosen = 0
    for index, level in enumerate(levels):
        if level0_um * levels[0].shape[1] / level.shape[1] <= target_um:
            chosen = index
    return chosen


def sidecar(slides: Path, stem: str) -> dict:
    """A slide's sidecar, named <stem>.json or <stem>.ome.json."""
    for name in (f"{stem}.json", f"{stem}.ome.json"):
        if (slides / name).exists():
            return json.loads((slides / name).read_text())
    raise FileNotFoundError(f"no sidecar for {stem} in {slides}")


def necrosis_field(scores: Path, he: list, level: int, top_left: tuple[int, int], size: int) -> np.ndarray:
    """Held-out necrosis over a square field of H&E level `level`: 0 surviving, 1 necrotic, 2 no tissue.

    Scores come from overlapping tiles (ml/predict_field.py); each pixel blends the four
    nearest tile centres, counting only those with tissue.
    """
    record = json.loads(scores.read_text())
    field = record["field"]
    probability = np.zeros(field["grid_shape"], np.float32)
    present = np.zeros(field["grid_shape"], np.float32)
    for t in record["tiles"]:
        probability[t["row"], t["col"]] = t["predicted"]
        present[t["row"], t["col"]] = 1.0

    # zoom pixels to positions on the tile-centre grid, through level 0
    to_field = he[field["level"]].shape[1] / he[level].shape[1]
    rows = ((top_left[0] + np.arange(size)) * to_field - field["first_centre"][0]) / field["stride_px"]
    cols = ((top_left[1] + np.arange(size)) * to_field - field["first_centre"][1]) / field["stride_px"]
    grid = np.meshgrid(rows, cols, indexing="ij")
    weight = map_coordinates(present, grid, order=1, mode="nearest")
    blended = map_coordinates(probability * present, grid, order=1, mode="nearest") / np.maximum(weight, 1e-6)
    necrotic = (blended >= record["cutoff"]).astype(np.uint8)
    return np.where(weight >= 0.5, necrotic, 2)


def cyp2e1_field(warped: list, he: list, um: float, spot: tuple[int, int]) -> tuple[np.ndarray, float]:
    """The relative CYP2E1 readout over the zoom field (0 negative, 1 positive, 2 excluded) and its um/px."""
    level = pick_level(he, um, READOUT_UM_PER_PX)
    factor = he[0].shape[1] / he[level].shape[1]
    half = int((ZOOM_UM / 2 + READOUT_MARGIN_UM) / (um * factor))
    r, c = int(spot[0] / factor), int(spot[1] / factor)
    top, left = max(r - half, 0), max(c - half, 0)
    bottom = min(r + half, he[level].shape[0])
    right = min(c + half, he[level].shape[1])
    rgb = np.asarray(warped[level][top:bottom, left:right])

    coarse_index = outline_level(warped)
    coarse = np.asarray(warped[coarse_index])
    outline = mask_from_level(coarse)
    to_coarse = he[level].shape[1] / coarse.shape[1]
    outline_crop = outline[int(top / to_coarse):int(np.ceil(bottom / to_coarse)),
                           int(left / to_coarse):int(np.ceil(right / to_coarse))]
    positive, _, _ = relative_readout(rgb, glass_colour(coarse, outline), outline_crop, um * factor)

    zoom = int(ZOOM_UM / 2 / (um * factor))
    return positive[r - top - zoom:r - top + zoom, c - left - zoom:c - left + zoom], um * factor


def read_block(slides: Path, field_scores: Path, block: str, spot: tuple[int, int]) -> dict:
    """The whole H&E thumbnail, the zooms in H&E and registered CYP2E1, and both segmentations."""
    he_info, he = read_pyramid(slides / f"mouse_apap_{block}_he.ome.tiff")
    _, cyp = read_pyramid(slides / f"mouse_apap_{block}_cyp2e1.ome.tiff")
    registration = Registration(**sidecar(slides, f"mouse_apap_{block}_cyp2e1")["registration"])
    field = load_field(slides, registration)
    if field is None:
        raise RuntimeError(f"{block}: no valid non-rigid field")
    um = he_info.pixel_size_um
    warped = nonrigid_levels(cyp, [level.shape[:2] for level in he], registration, field, um)

    thumb_level = max(i for i, level in enumerate(he) if max(level.shape[:2]) >= THUMB_EDGE_PX)
    thumb = np.asarray(he[thumb_level]).copy()
    tissue = mask_from_level(thumb)
    thumb[~tissue] = 255
    thumb_factor = he[0].shape[1] / he[thumb_level].shape[1]
    # cropped to the tissue and centred on a square, so every slide fills its panel
    rows, cols = np.nonzero(tissue)
    top, left = rows.min(), cols.min()
    height, width = rows.max() + 1 - top, cols.max() + 1 - left
    side = int(max(height, width) * THUMB_PAD)
    shift_r, shift_c = (side - height) // 2 - top, (side - width) // 2 - left
    canvas = np.full((side, side, 3), 255, np.uint8)
    canvas[shift_r + top:shift_r + top + height, shift_c + left:shift_c + left + width] = \
        thumb[top:top + height, left:left + width]

    level = pick_level(he, um, ZOOM_UM_PER_PX)
    factor = he[0].shape[1] / he[level].shape[1]
    half = int(ZOOM_UM / 2 / (um * factor))
    r, c = int(spot[0] / factor), int(spot[1] / factor)
    window = np.s_[r - half:r + half, c - half:c + half]
    segmentation, segmentation_um = cyp2e1_field(warped, he, um, spot)
    return {
        "thumb": canvas,
        "thumb_um_per_px": um * thumb_factor,
        "box": ((spot[1] - ZOOM_UM / 2 / um) / thumb_factor + shift_c,
                (spot[0] - ZOOM_UM / 2 / um) / thumb_factor + shift_r,
                ZOOM_UM / um / thumb_factor),
        "he": np.asarray(he[level][window]),
        "necrosis": necrosis_field(field_scores / f"{block}_field.json", he, level,
                                   (r - half, c - half), 2 * half),
        "cyp": np.asarray(warped[level][window].compute()),
        "cyp_segmentation": segmentation,
        "segmentation_um_per_px": segmentation_um,
        "zoom_um_per_px": um * factor,
    }


def scale_bar(ax, um_per_px: float, length_um: float, label: str) -> None:
    """A black bar in the lower right corner."""
    height, width = ax.get_images()[0].get_array().shape[:2]
    length = length_um / um_per_px
    x = width * 0.9 - length
    y = height * 0.93
    ax.plot([x, x + length], [y, y], color="black", lw=1.2, solid_capstyle="butt")
    ax.text(x + length / 2, y - height * 0.03, label, ha="center", va="bottom", fontsize=5)


def area_panel(ax, results: Path) -> None:
    """Held-out necrotic area per animal, boxed by dose, shown animals filled."""
    by_dose: dict[int, list[tuple[str, float]]] = {}
    for fold in json.loads(results.read_text())["folds"]:
        dose = int(fold["held_out"][:3])
        by_dose.setdefault(dose, []).append((fold["held_out"], 100 * fold["area_predicted"]))

    doses = sorted(by_dose)
    ax.boxplot([[v for _, v in by_dose[d]] for d in doses], positions=doses, widths=35,
               showfliers=False, medianprops={"color": "black"},
               boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8},
               capprops={"linewidth": 0.8})
    rng = np.random.default_rng(0)
    for dose in doses:
        for animal, value in by_dose[dose]:
            chosen = animal == BLOCKS[dose][0]
            ax.scatter(dose + rng.uniform(-10, 10), value, s=10, zorder=3,
                       facecolor="black" if chosen else "white", edgecolor="black", lw=0.6)
    ax.set_xlim(-40, 540)
    ax.set_xticks(doses)
    ax.set_xlabel("APAP dose [mg/kg]", fontsize=7)
    ax.set_ylabel("Necrotic area [% of tissue]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)


def overlay(ax, image: np.ndarray, mask: np.ndarray, colour: str) -> None:
    """A stain with a translucent fill and an outline where `mask` is set, stretched to the image."""
    h, w = image.shape[:2]
    extent = (-0.5, w - 0.5, h - 0.5, -0.5)
    ax.imshow(image)
    fill = np.zeros((*mask.shape, 4), np.float32)
    fill[mask] = (*matplotlib.colors.to_rgb(colour), OVERLAY_ALPHA)
    ax.imshow(fill, extent=extent, interpolation="nearest")
    ys = np.linspace(0, h - 1, mask.shape[0])
    xs = np.linspace(0, w - 1, mask.shape[1])
    ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=colour, linewidths=0.5)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def image_panel(ax, data: dict, row: int, style: str) -> None:
    """One cell of the image grid; the model rows stay empty until simulations exist."""
    if row == 0:
        ax.imshow(data["thumb"])
        x, y, size = data["box"]
        ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="black", lw=0.6))
    elif row == 1:
        ax.imshow(data["he"])
    elif row == 2 and style == "overlay":
        overlay(ax, data["he"], data["necrosis"] == 1, NECROSIS_OVERLAY)
    elif row == 2:
        ax.imshow(data["necrosis"], cmap=NECROSIS_COLOURS, vmin=0, vmax=2, interpolation="nearest")
    elif row == 3:
        ax.imshow(data["cyp"])
    elif row == 4 and style == "overlay":
        overlay(ax, data["cyp"], data["cyp_segmentation"] == 1, CYP2E1_OVERLAY)
    elif row == 4:
        ax.imshow(data["cyp_segmentation"], cmap=CYP2E1_COLOURS, vmin=0, vmax=2,
                  interpolation="nearest")
    else:
        ax.set_facecolor("#f2f2f2")
        ax.text(0.5, 0.5, "placeholder", ha="center", va="center",
                transform=ax.transAxes, fontsize=5, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def draw(blocks: dict, style: str, results: Path, out: Path) -> None:
    """The whole figure in one style, panels placed in millimetres so the grid stays tight."""
    rows, columns = len(ROWS), len(blocks)
    fixed = TITLE_MM + (rows - 1) * GAP_MM + PLOT_GAP_MM + PLOT_MM + BOTTOM_MM
    panel = (HEIGHT_MM - fixed) / rows
    grid_width = columns * panel + (columns - 1) * GAP_MM
    width = LABEL_MM + grid_width + RIGHT_MM
    fig = plt.figure(figsize=(width * MM, HEIGHT_MM * MM))

    def rect(left: float, top: float, w: float, h: float) -> list[float]:
        """A rectangle in mm from the top-left corner, as figure fractions."""
        return [left / width, (HEIGHT_MM - top - h) / HEIGHT_MM, w / width, h / HEIGHT_MM]

    for col, (dose, data) in enumerate(blocks.items()):
        left = LABEL_MM + col * (panel + GAP_MM)
        fig.text((left + panel / 2) / width, (HEIGHT_MM - TITLE_MM + 1.5) / HEIGHT_MM,
                 f"{dose} mg/kg", ha="center", va="bottom", fontsize=7)
        for row, label in enumerate(ROWS):
            ax = fig.add_axes(rect(left, TITLE_MM + row * (panel + GAP_MM), panel, panel))
            image_panel(ax, data, row, style)
            if row == 0:  # slides are scaled to fill their squares, so each needs its own bar
                scale_bar(ax, data["thumb_um_per_px"], 2000, "2 mm")
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
                if row in (1, 2, 3):
                    scale_bar(ax, data["zoom_um_per_px"], 200, "200 µm")
                elif row == 4:
                    um_per_px = data["zoom_um_per_px" if style == "overlay" else "segmentation_um_per_px"]
                    scale_bar(ax, um_per_px, 200, "200 µm")

    plot_top = TITLE_MM + rows * panel + (rows - 1) * GAP_MM + PLOT_GAP_MM
    area_panel(fig.add_axes(rect(LABEL_MM, plot_top, grid_width, PLOT_MM)), results)

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"necrosis_figure_{style}.{suffix}", dpi=400)
    plt.close(fig)
    print(f"{style}: {width:.0f} x {HEIGHT_MM} mm, panels {panel:.1f} mm")


def main() -> None:
    """Read the six blocks once and draw the figure in both styles."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--field-scores", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    blocks = {dose: read_block(args.slides, args.field_scores, block, spot)
              for dose, (block, spot) in BLOCKS.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        draw(blocks, style, args.results, args.out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
