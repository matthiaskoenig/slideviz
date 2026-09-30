"""Manuscript figure: whole H&E per dose, a zoom in H&E and CYP2E1, necrotic area by dose.

    uv run --with matplotlib python scripts/manuscript_figure.py \
        --slides <APAP_tiff> --results <necrosis_results_nested_area.json> --out <figures>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from slideviz.analysis.tissue import mask_from_level
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
THUMB_EDGE_PX = 1200

MM = 1 / 25.4
WIDTH_MM = 190  # Elsevier full page width


def pick_level(levels, level0_um: float, target_um: float) -> int:
    """The coarsest level at or finer than `target_um`."""
    chosen = 0
    for index, level in enumerate(levels):
        if level0_um * levels[0].shape[1] / level.shape[1] <= target_um:
            chosen = index
    return chosen


def read_block(slides: Path, block: str, spot: tuple[int, int]) -> dict:
    """The whole H&E thumbnail and the zoom in H&E and registered CYP2E1."""
    he_info, he = read_pyramid(slides / f"mouse_apap_{block}_he.ome.tiff")
    _, cyp = read_pyramid(slides / f"mouse_apap_{block}_cyp2e1.ome.tiff")
    sidecar = json.loads((slides / f"mouse_apap_{block}_cyp2e1.json").read_text())
    registration = Registration(**sidecar["registration"])
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
    return {
        "thumb": thumb,
        "thumb_um_per_px": um * thumb_factor,
        "box": ((spot[1] - ZOOM_UM / 2 / um) / thumb_factor,
                (spot[0] - ZOOM_UM / 2 / um) / thumb_factor,
                ZOOM_UM / um / thumb_factor),
        "he": np.asarray(he[level][window]),
        "cyp": np.asarray(warped[level][window].compute()),
        "zoom_um_per_px": um * factor,
    }


def scale_bar(ax, um_per_px: float, length_um: float, label: str) -> None:
    """A black bar in the lower right corner."""
    height, width = ax.get_images()[0].get_array().shape[:2]
    length = length_um / um_per_px
    x = width * 0.95 - length
    y = height * 0.93
    ax.plot([x, x + length], [y, y], color="black", lw=1.5, solid_capstyle="butt")
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


def main() -> None:
    """Read the six blocks and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    blocks = {dose: read_block(args.slides, block, spot)
              for dose, (block, spot) in BLOCKS.items()}

    fig = plt.figure(figsize=(WIDTH_MM * MM, WIDTH_MM * MM * 1.05))
    # the fourth column is a spacer
    grid = fig.add_gridspec(len(BLOCKS), 5, width_ratios=[1.3, 1, 1, 0.45, 2.2],
                            wspace=0.05, hspace=0.05)
    for row, (dose, data) in enumerate(blocks.items()):
        axes = [fig.add_subplot(grid[row, col]) for col in range(3)]
        axes[0].imshow(data["thumb"])
        x, y, size = data["box"]
        axes[0].add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="black", lw=0.8))
        axes[1].imshow(data["he"])
        axes[2].imshow(data["cyp"])
        scale_bar(axes[0], data["thumb_um_per_px"], 2000, "2 mm")
        scale_bar(axes[1], data["zoom_um_per_px"], 200, "200 µm")
        scale_bar(axes[2], data["zoom_um_per_px"], 200, "200 µm")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
        axes[0].set_ylabel(f"{dose}", fontsize=7)
        if row == 0:
            for ax, title in zip(axes, ("H&E", "H&E", "CYP2E1"), strict=True):
                ax.set_title(title, fontsize=7)
    fig.text(0.1, 0.5, "APAP [mg/kg]", rotation=90, va="center", fontsize=7)

    area = fig.add_subplot(grid[1:5, 4])
    area_panel(area, args.results)
    area.set_box_aspect(1)

    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"necrosis_figure.{suffix}", dpi=400, bbox_inches="tight")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
