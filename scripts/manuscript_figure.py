"""Manuscript figure on one A4 page: per dose, whole slides, zooms, necrosis and the CYP2E1 gradient, model rows, area and time plots.

Picture rows use the flattening fields in <slides>/flatten/; the segmentations use raw pixels.
Necrosis is the necrosis map and its areas from the necrosis_slides.py run in --necrosis, vessel lumina left out.
The time plot takes the simulated curves from --model, the PI's time courses (one zonation pattern, doses D1 to D8).

    uv run --with matplotlib python scripts/manuscript_figure.py --slides <APAP_tiff> \
        --necrosis <necrosis_slides.py output> --model <necrosis-dose-timecourses.csv> --out <figures>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates
from skimage.color import lab2rgb, rgb2lab

from slideviz.analysis import flatten
from slideviz.analysis.dab import (
    STEP_ALPHA,
    STEP_OPACITY,
    STEP_RGB,
    glass_colour,
    gradient_steps,
    relative_position,
    relative_readout,
)
from slideviz.analysis.prediction import CALL_OPACITY, CALL_RGB, calls_on_level
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

ROWS = ("H&E", "CYP2E1", "H&E", "CYP2E1", "Necrosis\nsegmentation", "CYP2E1\ngradient,\nnecrosis",
        "Model\nCYP2E1", "Model\nnecrosis")
# the gradient steps' colours (analysis/dab.py) per step value, step 0 and no tissue clear
STEP_RGBA = np.array([(0.0, 0.0, 0.0, 0.0)]
                     + [(*(c / 255 for c in rgb), alpha * STEP_OPACITY)
                        for rgb, alpha in zip(STEP_RGB, STEP_ALPHA, strict=True)]
                     + [(0.0, 0.0, 0.0, 0.0)], np.float32)
NECROSIS_RECORD = "necrosis_slides.json"  # the necrosis run's areas, beside its maps

# page layout in mm: one A4 page, the panel size follows from the room the rows leave
MM = 1 / 25.4
PAGE_MM = (210.0, 297.0)
MARGIN_MM = 8.0
LABEL_MM = 14.0  # row labels left of the grid
GAP_MM = 1.0
TITLE_MM = 6.0
KEY_GAP_MM = 2.0
KEY_MM = 5.0  # colour key of the gradient row, under the grid
PLOT_GAP_MM = 9.0
PLOT_MM = 38.0
PLOT_SPACE_MM = 14.0  # between the two plots, room for the right one's axis label
BOTTOM_MM = 10.0

# necrosis against time: the slides were taken at one time point, the simulation runs to it
DATA_HOURS = 24
TIME_MAX_H = 26  # just past the data, whose markers spread around 24 h
TIME_TICK_H = 6
DOSE_SPREAD_H = 0.7  # markers per dose spread around the time point so error bars stay apart
# the simulation (PI, 21 Oct): this zonation pattern; its doses D1 to D8 are not matched to mg/kg, so greys, not blues
MODEL_PATTERN = "Linear increase"
MODEL_LIGHT, MODEL_DARK = (200, 200, 200), (40, 40, 40)
# doses light to dark in the necrosis blue (the call's cyan as it shows over the H&E), at these CIELAB lightnesses;
# the lightest is 2.4:1 against white
DOSE_LIGHTNESS = (68.0, 28.0)


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


# the overlapping-tile scores of the 18 Oct comparison (scripts/necrosis_fine_report.py imports this)
def field_score(scores: Path, he: list, level: int, top_left: tuple[int, int], size: int) -> tuple:
    """The blended held-out score over a square field, the share of tissue tiles behind it, and the cutoff."""
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
    return blended, weight, record["cutoff"]


def cyp2e1_field(warped: list, he: list, um: float, spot: tuple[int, int]) -> tuple[np.ndarray, float]:
    """The CYP2E1 gradient steps over the zoom field (0 to STEPS - 1, STEPS excluded) and its um/px."""
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
    positive, smoothed, levels = relative_readout(rgb, glass_colour(coarse, outline), outline_crop, um * factor)
    steps = gradient_steps(relative_position(smoothed, positive != 2, levels))

    zoom = int(ZOOM_UM / 2 / (um * factor))
    return steps[r - top - zoom:r - top + zoom, c - left - zoom:c - left + zoom], um * factor


def flattening(slides: Path, stem: str) -> flatten.Flattening:
    """A slide's flattening fields from the folder beside it."""
    path = slides / "flatten" / f"{stem}_flatten.npz"
    if not path.exists():
        raise FileNotFoundError(f"no flattening fields: {path}")
    return flatten.load(path)


def square(image: np.ndarray, tissue: np.ndarray, box: tuple[int, int, int, int], side: int) -> np.ndarray:
    """The tissue inside box (top, left, height, width) centred on a white square of edge `side`."""
    top, left, height, width = box
    canvas = np.full((side, side, 3), 255, np.uint8)
    shown = np.where(tissue[..., None], image, 255)[top:top + height, left:left + width]
    row, col = (side - height) // 2, (side - width) // 2
    canvas[row:row + height, col:col + width] = shown
    return canvas


def read_block(slides: Path, necrosis_dir: Path, block: str, spot: tuple[int, int]) -> dict:
    """Whole H&E and CYP2E1 thumbnails, the zooms in H&E and registered CYP2E1, necrosis and gradient."""
    he_info, he = read_pyramid(slides / f"mouse_apap_{block}_he.ome.tiff")
    cyp_info, cyp = read_pyramid(slides / f"mouse_apap_{block}_cyp2e1.ome.tiff")
    registration = Registration(**sidecar(slides, f"mouse_apap_{block}_cyp2e1")["registration"])
    field = load_field(slides, registration)
    if field is None:
        raise RuntimeError(f"{block}: no valid non-rigid field")
    um = he_info.pixel_size_um
    he_shapes = [level.shape[:2] for level in he]
    warped = nonrigid_levels(cyp, he_shapes, registration, field, um)
    # the picture rows show the slides with the staining gradients divided out
    he_shown = flatten.flatten_levels(he, flattening(slides, f"mouse_apap_{block}_he"), um)
    cyp_flat = flatten.flatten_levels(cyp, flattening(slides, f"mouse_apap_{block}_cyp2e1"),
                                      cyp_info.pixel_size_um)
    cyp_shown = nonrigid_levels(cyp_flat, he_shapes, registration, field, um)

    thumb_level = max(i for i, level in enumerate(he) if max(level.shape[:2]) >= THUMB_EDGE_PX)
    he_tissue = mask_from_level(np.asarray(he[thumb_level]))
    cyp_tissue = mask_from_level(np.asarray(warped[thumb_level]))
    thumb_factor = he[0].shape[1] / he[thumb_level].shape[1]
    # both sections on one square in the H&E frame, cropped to their joint extent, so the thumbnails line up
    rows, cols = np.nonzero(he_tissue | cyp_tissue)
    top, left = rows.min(), cols.min()
    height, width = rows.max() + 1 - top, cols.max() + 1 - left
    side = int(max(height, width) * THUMB_PAD)
    shift_r, shift_c = (side - height) // 2 - top, (side - width) // 2 - left
    box = (top, left, height, width)

    level = pick_level(he, um, ZOOM_UM_PER_PX)
    factor = he[0].shape[1] / he[level].shape[1]
    half = int(ZOOM_UM / 2 / (um * factor))
    r, c = int(spot[0] / factor), int(spot[1] / factor)
    window = np.s_[r - half:r + half, c - half:c + half]
    steps, steps_um = cyp2e1_field(warped, he, um, spot)
    # 0 surviving, 1 necrotic, 2 not measured (glass, vessel lumina)
    necrotic, measured, _ = calls_on_level(necrosis_dir / f"{block}_necrosis.npz", he_shapes, level)
    necrosis = np.where(measured, necrotic, 2).astype(np.uint8)[window]
    return {
        "he_thumb": square(np.asarray(he_shown[thumb_level]), he_tissue, box, side),
        "cyp_thumb": square(np.asarray(cyp_shown[thumb_level]), cyp_tissue, box, side),
        "thumb_um_per_px": um * thumb_factor,
        "box": ((spot[1] - ZOOM_UM / 2 / um) / thumb_factor + shift_c,
                (spot[0] - ZOOM_UM / 2 / um) / thumb_factor + shift_r,
                ZOOM_UM / um / thumb_factor),
        "he": np.asarray(he_shown[level][window]),
        "necrosis": necrosis,
        "cyp": np.asarray(cyp_shown[level][window].compute()),
        "cyp_steps": steps,
        "steps_um_per_px": steps_um,
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


def necrotic_areas(necrosis_dir: Path) -> dict[int, list[tuple[str, float]]]:
    """Necrotic area of the necrosis map without lumina in percent, per animal, grouped by dose."""
    by_dose: dict[int, list[tuple[str, float]]] = {}
    for row in sorted(json.loads((necrosis_dir / NECROSIS_RECORD).read_text())["slides"], key=lambda r: r["animal"]):
        by_dose.setdefault(int(row["animal"][:3]), []).append((row["animal"], 100 * row["area_E"]))
    return dict(sorted(by_dose.items()))


def dose_colours(doses: list[int], base: np.ndarray) -> dict[int, tuple]:
    """One blue per dose, light to dark, in the hue the necrotic call has over the H&E `base`, so plots match panels."""
    shown = CALL_OPACITY * np.array(CALL_RGB) + (1 - CALL_OPACITY) * base
    _, a, b = rgb2lab(shown[None, None])[0, 0]
    return {dose: tuple(np.clip(lab2rgb(np.array([[[lightness, a, b]]]))[0, 0], 0, 1))
            for dose, lightness in zip(doses, np.linspace(*DOSE_LIGHTNESS, len(doses)), strict=True)}


def area_panel(ax, by_dose: dict, top: float) -> None:
    """Necrotic area per animal, boxed by dose with a line through the medians, shown animals filled."""
    doses = list(by_dose)
    values = [[v for _, v in by_dose[d]] for d in doses]
    ax.boxplot(values, positions=doses, widths=35,
               showfliers=False, medianprops={"color": "black"},
               boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8},
               capprops={"linewidth": 0.8})
    ax.plot(doses, [np.median(v) for v in values], color="black", lw=0.8, zorder=2)
    rng = np.random.default_rng(0)
    for dose in doses:
        for animal, value in by_dose[dose]:
            chosen = animal == BLOCKS[dose][0]
            ax.scatter(dose + rng.uniform(-10, 10), value, s=10, zorder=3,
                       facecolor="black" if chosen else "white", edgecolor="black", lw=0.6)
    ax.set_xlim(-40, 540)
    ax.set_ylim(-2, top)
    ax.set_xticks(doses)
    ax.set_xlabel("APAP dose [mg/kg]", fontsize=7)
    ax.set_ylabel("Necrotic area [% of tissue]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)


def read_model(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The simulated curves of MODEL_PATTERN per dose label, as hours and percent of tissue necrotic."""
    curves: dict[tuple[int, str], list] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["pattern"].split(maxsplit=1)[1] == MODEL_PATTERN:  # "B  Linear increase"
                curves.setdefault((int(row["dose"]), row["dose_label"]), []).append(
                    (float(row["time_h"]), float(row["necrosis_percent"])))
    if not curves:
        raise ValueError(f"no pattern {MODEL_PATTERN!r} in {path}")
    return {label: tuple(np.array(sorted(points)).T) for (_, label), points in sorted(curves.items())}


def time_panel(ax, by_dose: dict, top: float, base: np.ndarray, model: dict | None = None) -> None:
    """Necrotic area against time after dosing: the data as mean and SD per dose, the simulated curves in grey."""
    if model:
        for (t, y), share in zip(model.values(), np.linspace(0, 1, len(model)), strict=True):
            grey = [((1 - share) * a + share * b) / 255 for a, b in zip(MODEL_LIGHT, MODEL_DARK, strict=True)]
            ax.plot(t, y, color=grey, lw=0.8, zorder=1)
        first, last = next(iter(model)), list(model)[-1]
        ax.text(0.03, 0.97, f"model, {MODEL_PATTERN.lower()}: {first} light to {last} dark", transform=ax.transAxes,
                ha="left", va="top", fontsize=5, color="#555555")
    else:
        ax.text(0.03, 0.97, "model curves: placeholder", transform=ax.transAxes, ha="left", va="top",
                fontsize=5, color="#888888")
    colours = dose_colours(list(by_dose), base)
    middle = (len(by_dose) - 1) / 2
    for k, (dose, animals) in enumerate(by_dose.items()):
        values = np.array([v for _, v in animals])
        ax.errorbar(DATA_HOURS + (k - middle) * DOSE_SPREAD_H, values.mean(), yerr=values.std(ddof=1),
                    fmt="o", ms=3, color=colours[dose], ecolor=colours[dose], elinewidth=0.8, capsize=1.5,
                    label=f"{dose} mg/kg", zorder=3)
    ax.set_xlim(0, TIME_MAX_H)
    ax.set_ylim(-2, top)
    ax.set_xticks(range(0, DATA_HOURS + 1, TIME_TICK_H))
    ax.set_xlabel("Time after APAP [h]", fontsize=7)
    ax.set_ylabel("Necrotic area [% of tissue]", fontsize=7)
    ax.tick_params(labelsize=6)
    # above the plot, where the simulated curves leave no room; matplotlib fills columns first, so the doses
    # are reordered to read across the two rows
    handles, labels = ax.get_legend_handles_labels()
    columns = 3
    rows = -(-len(handles) // columns)
    order = [r * columns + c for c in range(columns) for r in range(rows) if r * columns + c < len(handles)]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=5, frameon=False, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), ncol=columns, handlelength=1, columnspacing=1.2, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def key_panel(ax, base: np.ndarray) -> None:
    """Colour key of the gradient row, each swatch mixed over the H&E colour as in the panels."""
    def over(rgba: np.ndarray, below: np.ndarray) -> np.ndarray:
        """One see-through colour laid over another."""
        return rgba[3] * rgba[:3] + (1 - rgba[3]) * below

    call = np.array([*CALL_RGB, CALL_OPACITY])
    swatches = [([over(STEP_RGBA[step], base) for step in range(1, len(STEP_RGB) + 1)], "CYP2E1, local low to high"),
                ([over(call, base)], "necrotic"),
                ([over(call, over(STEP_RGBA[len(STEP_RGB)], base))], "necrotic and CYP2E1 high")]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    x, width = 0.0, 0.03
    for colours, label in swatches:
        for colour in colours:
            ax.add_patch(Rectangle((x, 0.15), width, 0.7, facecolor=colour, edgecolor="none"))
            x += width
        ax.text(x + 0.008, 0.5, label, va="center", fontsize=6)
        x += 0.008 + 0.0075 * len(label) + 0.03


def area_overlay(ax, image: np.ndarray, necrotic: np.ndarray, steps: np.ndarray | None = None) -> None:
    """See-through areas on a stain as the viewer draws them: CYP2E1 steps in gold when given, the necrotic call in cyan on top."""
    h, w = image.shape[:2]
    extent = (-0.5, w - 0.5, h - 0.5, -0.5)
    ax.imshow(image)
    if steps is not None:
        ax.imshow(STEP_RGBA[steps], extent=extent, interpolation="nearest")
    call = np.zeros((*necrotic.shape, 4), np.float32)
    call[necrotic] = (*CALL_RGB, CALL_OPACITY)
    ax.imshow(call, extent=extent, interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def image_panel(ax, data: dict, row: int) -> None:
    """One cell of the image grid; the model rows stay empty until simulations exist."""
    if row in (0, 1):
        ax.imshow(data["he_thumb" if row == 0 else "cyp_thumb"])
        x, y, size = data["box"]
        ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="black", lw=0.6))
    elif row == 2:
        ax.imshow(data["he"])
    elif row == 3:
        ax.imshow(data["cyp"])
    elif row == 4:
        area_overlay(ax, data["he"], data["necrosis"] == 1)
    elif row == 5:
        area_overlay(ax, data["he"], data["necrosis"] == 1, data["cyp_steps"])
    else:
        ax.set_facecolor("#f2f2f2")
        ax.text(0.5, 0.5, "placeholder", ha="center", va="center",
                transform=ax.transAxes, fontsize=5, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def draw(blocks: dict, necrosis_dir: Path, out: Path, model: dict | None = None) -> None:
    """The whole figure on one A4 page, panels placed in millimetres so the grid stays tight."""
    rows, columns = len(ROWS), len(blocks)
    page_width, page_height = PAGE_MM
    fixed = TITLE_MM + (rows - 1) * GAP_MM + KEY_GAP_MM + KEY_MM + PLOT_GAP_MM + PLOT_MM + BOTTOM_MM
    by_height = (page_height - 2 * MARGIN_MM - fixed) / rows
    by_width = (page_width - 2 * MARGIN_MM - LABEL_MM - (columns - 1) * GAP_MM) / columns
    panel = min(by_height, by_width)
    grid_width = columns * panel + (columns - 1) * GAP_MM
    # the grid and its labels centred across the page
    origin = (page_width - LABEL_MM - grid_width) / 2 + LABEL_MM
    fig = plt.figure(figsize=(page_width * MM, page_height * MM))

    def rect(left: float, top: float, w: float, h: float) -> list[float]:
        """A rectangle in mm from the page's top-left corner, as figure fractions."""
        return [left / page_width, (page_height - top - h) / page_height, w / page_width, h / page_height]

    for col, (dose, data) in enumerate(blocks.items()):
        left = origin + col * (panel + GAP_MM)
        fig.text((left + panel / 2) / page_width, (page_height - MARGIN_MM - TITLE_MM + 1.5) / page_height,
                 f"{dose} mg/kg", ha="center", va="bottom", fontsize=7)
        for row, label in enumerate(ROWS):
            ax = fig.add_axes(rect(left, MARGIN_MM + TITLE_MM + row * (panel + GAP_MM), panel, panel))
            image_panel(ax, data, row)
            if row in (0, 1):  # slides are scaled to fill their squares, so each needs its own bar
                scale_bar(ax, data["thumb_um_per_px"], 2000, "2 mm")
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
                if row in (2, 3, 4, 5):
                    scale_bar(ax, data["zoom_um_per_px"], 200, "200 µm")

    grid_bottom = MARGIN_MM + TITLE_MM + rows * panel + (rows - 1) * GAP_MM
    # the key's swatches are mixed over the plain H&E of the first dose, the colour the gradient row lies on
    first = next(iter(blocks.values()))["he"].reshape(-1, 3)
    base = np.median(first[first.mean(axis=1) < 230], axis=0) / 255
    key_panel(fig.add_axes(rect(origin, grid_bottom + KEY_GAP_MM, grid_width, KEY_MM)), base)

    by_dose = necrotic_areas(necrosis_dir)
    highest = max([v for animals in by_dose.values() for _, v in animals]
                  + [float(y.max()) for _, y in (model or {}).values()])
    top = 1.08 * highest + 2  # both plots on one scale
    plot_top = grid_bottom + KEY_GAP_MM + KEY_MM + PLOT_GAP_MM
    plot_width = (grid_width - PLOT_SPACE_MM) / 2
    area_panel(fig.add_axes(rect(origin, plot_top, plot_width, PLOT_MM)), by_dose, top)
    time_panel(fig.add_axes(rect(origin + plot_width + PLOT_SPACE_MM, plot_top, plot_width, PLOT_MM)), by_dose, top,
               base, model)

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"necrosis_figure.{suffix}", dpi=400)
    plt.close(fig)
    print(f"A4 {page_width:.0f} x {page_height:.0f} mm, grid {LABEL_MM + grid_width:.0f} mm wide, panels {panel:.1f} mm")


def main() -> None:
    """Read the six blocks once and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--necrosis", type=Path, required=True, help="necrosis_slides.py output")
    parser.add_argument("--model", type=Path, help="the PI's simulated necrosis time courses (csv)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = read_model(args.model) if args.model else None

    blocks = {dose: read_block(args.slides, args.necrosis, block, spot)
              for dose, (block, spot) in BLOCKS.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    draw(blocks, args.necrosis, args.out, model)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
