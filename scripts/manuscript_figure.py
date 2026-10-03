"""Manuscript figure: per dose, whole H&E, zooms with necrosis and CYP2E1 segmentation, model rows, necrotic area.

    uv run --with matplotlib python scripts/manuscript_figure.py --slides <APAP_tiff> \
        --predictions <nested predictions> --results <necrosis_results_nested_area.json> --out <figures>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

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
    158: ("158mg_m2", (8670, 29070)),
    281: ("281mg_m1", (4250, 12750)),
    375: ("375mg_m3", (2890, 19210)),
    500: ("500mg_m1", (4250, 19890)),
}

ZOOM_UM = 1000.0
ZOOM_UM_PER_PX = 0.9
READOUT_UM_PER_PX = 1.8  # the CYP2E1 readout's working resolution
READOUT_MARGIN_UM = 600.0  # more than one readout window, so the field matches the whole-slide run
THUMB_EDGE_PX = 1200

ROWS = ("H&E", "H&E", "Necrosis\nsegmentation", "CYP2E1", "CYP2E1\nsegmentation",
        "Model\nCYP2E1", "Model\nnecrosis")
# negative or surviving, positive or necrotic, outside tissue
NECROSIS_COLOURS = ListedColormap(["#ececf4", "#c0392b", "white"])
CYP2E1_COLOURS = ListedColormap(["#ececf4", "#8b4513", "white"])

MM = 1 / 25.4
WIDTH_MM = 190  # Elsevier full page width
HEIGHT_MM = 245


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


def necrosis_field(prediction: Path, he: list, level: int, top_left: tuple[int, int], size: int) -> np.ndarray:
    """Nested necrosis tiles over a square field of H&E level `level`: 0 surviving, 1 necrotic, 2 untiled."""
    record = json.loads(prediction.read_text())
    tiles = record["tiles"]
    grid = np.full((max(t["row"] for t in tiles) + 1, max(t["col"] for t in tiles) + 1), 2, np.uint8)
    for t in tiles:
        grid[t["row"], t["col"]] = t["predicted"] >= record["cutoff"]
    # field pixels to tile indices, through the prediction's own level
    scale = he[record["level"]].shape[1] / he[level].shape[1] / record["size_px"]
    rows = ((top_left[0] + np.arange(size)) * scale).astype(int)
    cols = ((top_left[1] + np.arange(size)) * scale).astype(int)
    inside = (rows[:, None] < grid.shape[0]) & (cols[None, :] < grid.shape[1])
    field = grid[np.minimum(rows, grid.shape[0] - 1)][:, np.minimum(cols, grid.shape[1] - 1)]
    return np.where(inside, field, 2)


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


def read_block(slides: Path, predictions: Path, block: str, spot: tuple[int, int]) -> dict:
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
    thumb[~mask_from_level(thumb)] = 255
    thumb_factor = he[0].shape[1] / he[thumb_level].shape[1]

    level = pick_level(he, um, ZOOM_UM_PER_PX)
    factor = he[0].shape[1] / he[level].shape[1]
    half = int(ZOOM_UM / 2 / (um * factor))
    r, c = int(spot[0] / factor), int(spot[1] / factor)
    window = np.s_[r - half:r + half, c - half:c + half]
    segmentation, segmentation_um = cyp2e1_field(warped, he, um, spot)
    return {
        "thumb": thumb,
        "thumb_um_per_px": um * thumb_factor,
        "box": ((spot[1] - ZOOM_UM / 2 / um) / thumb_factor,
                (spot[0] - ZOOM_UM / 2 / um) / thumb_factor,
                ZOOM_UM / um / thumb_factor),
        "he": np.asarray(he[level][window]),
        "necrosis": necrosis_field(predictions / f"{block}_predictions.json", he, level,
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
    x = width * 0.95 - length
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


def image_panel(ax, data: dict, row: int) -> None:
    """One cell of the image grid; the model rows stay empty until simulations exist."""
    if row == 0:
        ax.imshow(data["thumb"])
        x, y, size = data["box"]
        ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="black", lw=0.6))
    elif row == 1:
        ax.imshow(data["he"])
    elif row == 2:
        ax.imshow(data["necrosis"], cmap=NECROSIS_COLOURS, vmin=0, vmax=2, interpolation="nearest")
    elif row == 3:
        ax.imshow(data["cyp"])
    elif row == 4:
        ax.imshow(data["cyp_segmentation"], cmap=CYP2E1_COLOURS, vmin=0, vmax=2,
                  interpolation="nearest")
    else:
        ax.set_facecolor("#f2f2f2")
        ax.set_box_aspect(1)
        ax.text(0.5, 0.5, "simulation\n(placeholder)", ha="center", va="center",
                transform=ax.transAxes, fontsize=5, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def main() -> None:
    """Read the six blocks and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    blocks = {dose: read_block(args.slides, args.predictions, block, spot)
              for dose, (block, spot) in BLOCKS.items()}

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[len(ROWS), 1.9], hspace=0.08)
    grid = outer[0].subgridspec(len(ROWS), len(BLOCKS), wspace=0.04, hspace=0.06)
    titles = []
    for col, (dose, data) in enumerate(blocks.items()):
        for row, label in enumerate(ROWS):
            ax = fig.add_subplot(grid[row, col])
            image_panel(ax, data, row)
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
                if row == 0:
                    scale_bar(ax, data["thumb_um_per_px"], 2000, "2 mm")
                elif row in (1, 2, 3):
                    scale_bar(ax, data["zoom_um_per_px"], 200, "200 µm")
                elif row == 4:
                    scale_bar(ax, data["segmentation_um_per_px"], 200, "200 µm")
            if row == 1:
                titles.append((ax, f"{dose} mg/kg"))

    # one line above the grid; the thumbnails differ in shape, so their own tops do not align
    top = grid[0, 0].get_position(fig).y1
    for ax, title in titles:
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, top + 0.008, title, ha="center", va="bottom", fontsize=7)

    # centred under the grid, so its aspect stays readable
    area = fig.add_subplot(outer[1].subgridspec(1, 3, width_ratios=[1, 2.2, 1])[0, 1])
    area_panel(area, args.results)

    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"necrosis_figure.{suffix}", dpi=400, bbox_inches="tight")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
