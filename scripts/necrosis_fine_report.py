"""Necrosis maps side by side on the comparison fields: one PDF and a flip page, to judge them by eye.

    uv run --no-sync --with matplotlib python scripts/necrosis_fine_report.py --slides /data/michelle/slides \
        --fields <pick_fields.py output> --tiles /data/michelle/slides/predictions \
        --overlap <predict_field.py output> --patches <predict_field_patches.py output> \
        --overlap40 <predict_field.py output on 40 um tiles> --pixels <pixel_classifier.py output> \
        --lumen /data/michelle/necrosis_lumen_2026-10-11 \
        --annotations /data/michelle/slides/annotations/cvat_polygons.json \
        --overviews /data/michelle/slides/annotation_overviews --out <dir>

Every method's call is drawn as the viewer draws it, see-through cyan, and only on tissue (the outline
without lumina), the same mask for every method and for the areas. A field's first page shows the whole
1 mm field, its second a 250 um zoom at full resolution, placed where the methods disagree most. The flip
page shows the same views and switches methods with the number keys. DSC and IoU against the drawn
polygons are listed for information only: the polygons stop short of the necrosis edge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from manuscript_figure import field_score
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter

from slideviz.analysis.annotation import (
    cvat_polygons,
    rasterise,
    read_overview,
    to_level,
)
from slideviz.analysis.edge_snap import guided_filter
from slideviz.analysis.prediction import CALL_OPACITY, CALL_RGB, read_lumen
from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.io.reader import open_slide

ZOOM_UM = 250.0
PAGE = (1754, 1240)  # A4 landscape at 150 dpi
MARGIN = 40
FLIP_PX = 1100  # edge of each flip-page image

# edge snapping for E and F, chosen by eye on three fields (18 Oct): 8 and 16 um looked alike, 8 keeps more detail
SNAP_UM = 8.0
SNAP_EPS = 0.01

# pick_fields.py's reasons, short enough for the table
CHOSEN_BY = {"manuscript figure zoom": "figure zoom", "most necrosis edge": "most edge",
             "no necrosis edge, nearest the tissue centre": "tissue centre"}

METHODS = {
    "A": "today: one call per 75 um tile, on the tile grid",
    "B": "75 um tiles every 15 um, one call per tile",
    "C": "40 um tiles every 8 um, one call per tile (a 40 um model, trained like A's)",
    "D": "75 um tiles every 15 um, one call per 2.7 um patch, averaged over the tiles that see it",
    "E": f"D's score snapped to the H&E's edges (guided filter, {SNAP_UM:.0f} um window)",
    "F": f"B's score snapped to the H&E's edges (guided filter, {SNAP_UM:.0f} um window)",
    "G": "pixel classifier: colour, edge and texture measures per pixel, 0.4 to 7 um, and a random forest",
}


def font(size: int) -> ImageFont.FreeTypeFont:
    """matplotlib's bundled sans serif at a pixel size."""
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def pyramid_factor(fine_width: int, coarse_width: int) -> int:
    """The power of two between two pyramid levels: each level halves the one above."""
    return 2 ** round(np.log2(fine_width / coarse_width))


# Measured on the slides (18 Oct): a coarser level sits (factor - 1) / 2 level pixels up and left of a plain
# factor-wide box, the same across the slide, as halving by pyrDown gives; mapping by the ratio of the widths
# drifts from that by up to 13 level-1 px at the coarsest outline level
def on_level(mask: np.ndarray, factor: int, top: int, left: int, size: int) -> np.ndarray:
    """A square of a level's pixels on a coarser level's mask, coarse pixel q centred on level pixel factor * q."""
    rows = ((np.arange(top, top + size) + factor // 2) // factor).clip(0, mask.shape[0] - 1)
    cols = ((np.arange(left, left + size) + factor // 2) // factor).clip(0, mask.shape[1] - 1)
    return mask[np.ix_(rows, cols)]


def tile_calls(prediction: dict, top: int, left: int, size: int) -> np.ndarray:
    """Today's call per pixel: the call of the grid tile the pixel lies in, none where no tile was kept."""
    step = prediction["size_px"]
    shape = (max(t["row"] for t in prediction["tiles"]) + 1, max(t["col"] for t in prediction["tiles"]) + 1)
    called = np.zeros(shape, bool)
    for t in prediction["tiles"]:
        called[t["row"], t["col"]] = t["predicted"] >= prediction["cutoff"]
    rows, cols = (np.arange(top, top + size) // step), (np.arange(left, left + size) // step)
    inside = (rows < shape[0])[:, None] & (cols < shape[1])[None, :]
    return called[np.ix_(rows.clip(0, shape[0] - 1), cols.clip(0, shape[1] - 1))] & inside


def map_score(path: Path, key: str, top: int, left: int, size: int) -> tuple[np.ndarray, dict]:
    """A stored score map's value per pixel over a square, 0 where the map has none, and the map's record."""
    stored = np.load(path)
    meta = json.loads(str(stored["meta"]))
    origin = meta["origin_level_rc"]
    y, x = top - origin[0], left - origin[1]
    mean = stored[key]
    # no call where the field reaches past the map, as where no tile covered a pixel
    window = np.full((size, size), np.nan, np.float32)
    y0, x0, y1, x1 = max(0, y), max(0, x), min(mean.shape[0], y + size), min(mean.shape[1], x + size)
    if y1 > y0 and x1 > x0:
        window[y0 - y:y1 - y, x0 - x:x1 - x] = mean[y0:y1, x0:x1]
    return np.nan_to_num(window, nan=0.0), meta


def overlap_score(truth: np.ndarray, called: np.ndarray) -> tuple[float | None, float | None]:
    """DSC and IoU of two boolean maps, None where neither holds a positive."""
    both, drawn, kept = int((truth & called).sum()), int(truth.sum()), int(called.sum())
    if not drawn + kept:
        return None, None
    return round(2 * both / (drawn + kept), 4), round(both / (drawn + kept - both), 4)


def disagreement_centre(calls: dict, tissue: np.ndarray, window: int) -> tuple[int, int, bool]:
    """Centre of the window holding the most pixels the methods call differently, nearest their middle; or the field's."""
    stack = np.stack(list(calls.values()))
    differ = (tissue & (stack.any(axis=0) != stack.all(axis=0))).astype(np.float32)
    if not differ.any():
        return differ.shape[0] // 2, differ.shape[1] // 2, False
    density = uniform_filter(differ, size=window, mode="constant")
    half = window // 2
    inner = np.full(density.shape, -1.0, np.float32)
    inner[half:differ.shape[0] - half, half:differ.shape[1] - half] = density[half:differ.shape[0] - half,
                                                                              half:differ.shape[1] - half]
    rows, cols = np.nonzero(inner >= inner.max() * (1 - 1e-6))
    ys, xs = np.nonzero(differ)
    best = int(np.argmin(np.hypot(rows - ys.mean(), cols - xs.mean())))
    return int(rows[best]), int(cols[best]), True


def stretched(mask: np.ndarray, top: int, left: int, side: int, size: int) -> np.ndarray:
    """A square crop of a level-1 map, stretched onto the level-0 zoom of edge size."""
    crop = mask[top:top + side, left:left + side]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.NEAREST))


def tinted(rgb: np.ndarray, call: np.ndarray | None) -> np.ndarray:
    """The H&E with a call drawn over it in the viewer's see-through cyan."""
    if call is None:
        return rgb
    out = rgb.astype(np.float32)
    out[call] = (1 - CALL_OPACITY) * out[call] + CALL_OPACITY * 255 * np.array(CALL_RGB)
    return out.clip(0, 255).astype(np.uint8)


def page(title: str, panels: list[tuple[str, np.ndarray]], note: str) -> Image.Image:
    """One A4 page: a title, the panels in two rows, and a note at the foot."""
    sheet = Image.new("RGB", PAGE, (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((MARGIN, 18), title, fill=(0, 0, 0), font=font(26))
    columns = -(-len(panels) // 2)
    side = min((PAGE[0] - 2 * MARGIN - (columns - 1) * 16) // columns, (PAGE[1] - 160) // 2 - 34)
    for i, (label, image) in enumerate(panels):
        x = MARGIN + (i % columns) * (side + 16)
        y = 70 + (i // columns) * (side + 44)
        sheet.paste(Image.fromarray(image).resize((side, side), Image.LANCZOS), (x, y + 30))
        draw.text((x, y + 4), label, fill=(0, 0, 0), font=font(19))
    draw.text((MARGIN, PAGE[1] - 50), note, fill=(70, 70, 70), font=font(16))
    return sheet


def summary_page(rows: list[dict], record: dict) -> Image.Image:
    """The first page: what each method is, and every field's areas and overlap with the drawn polygons."""
    sheet = Image.new("RGB", PAGE, (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((MARGIN, 24), "Finer necrosis maps on 20 fields of 1 mm", fill=(0, 0, 0), font=font(30))
    y = 80
    for code, text in METHODS.items():
        draw.text((MARGIN, y), f"{code}   {text}", fill=(0, 0, 0), font=font(19))
        y += 28
    draw.text((MARGIN, y + 6), "Each map comes from a model trained without that animal; calls only on tissue (lumina "
              "out). DSC against your polygons, for information.", fill=(70, 70, 70), font=font(16))
    y += 50
    codes = list(METHODS)
    header = ["field", "chosen by", "drawn", *[f"{c} area" for c in codes], *[f"{c} DSC" for c in codes]]
    step = min(110, (PAGE[0] - 2 * MARGIN - 350) // (1 + 2 * len(codes)))
    widths = [150, 200, step, *[step] * len(codes), *[step] * len(codes)]
    xs = np.cumsum([MARGIN, *widths[:-1]])
    for x, text in zip(xs, header, strict=True):
        draw.text((int(x), y), text, fill=(0, 0, 0), font=font(18))
    y += 32
    for row in rows:
        cells = [row["animal"], CHOSEN_BY.get(row["chosen_by"], row["chosen_by"]), f"{row['area']['drawn']:.1%}",
                 *[f"{row['area'][c]:.1%}" for c in codes],
                 *["" if row["dsc"][c] is None else f"{row['dsc'][c]:.2f}" for c in codes]]
        for x, text in zip(xs, cells, strict=True):
            draw.text((int(x), y), text, fill=(0, 0, 0), font=font(17))
        y += 27
    means = record["mean_over_drawn_fields"]
    cells = ["mean, drawn", "", "", *[""] * len(codes),
             *["" if means["dsc"][c] is None else f"{means['dsc'][c]:.2f}" for c in codes]]
    for x, text in zip(xs, cells, strict=True):
        draw.text((int(x), y + 6), text, fill=(0, 0, 0), font=font(17))
    draw.text((MARGIN, PAGE[1] - 50), f"written {record['written']}, code {record['code']}", fill=(70, 70, 70),
              font=font(16))
    return sheet


FLIP_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Necrosis map flip</title>
<style>
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --on: #0a7f8c; }
@media (prefers-color-scheme: dark) { :root { --bg: #161617; --fg: #f5f5f7; --muted: #a1a1a6; --line: #3a3a3c; --on: #4cc9d6; } }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 15px/1.4 system-ui, sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
button, select { font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; }
button.on { border-color: var(--on); color: var(--on); }
img { max-width: 100%; height: auto; display: block; border: 1px solid var(--line); }
p { color: var(--muted); }
</style></head><body>
<header><select id="field"></select><span id="views"></span><span id="methods"></span></header>
<img id="view" alt="">
<p id="caption"></p>
<p id="help"></p>
<script>
const DATA = __DATA__;
let field = 0, view = "zoom", method = DATA.methods[0];
const $ = id => document.getElementById(id);
function draw() {
  const f = DATA.fields[field];
  $("field").value = field;
  $("view").src = `img/${f.animal}_${view}_${method}.jpg`;
  $("view").alt = `${f.animal}, ${view}, ${method}`;
  $("caption").textContent = `${f.animal}, ${view === "zoom" ? "250 um zoom" : "whole 1 mm field"}: ${DATA.names[method]}`;
  document.querySelectorAll("#methods button").forEach(b => b.classList.toggle("on", b.dataset.m === method));
  document.querySelectorAll("#views button").forEach(b => b.classList.toggle("on", b.dataset.v === view));
}
DATA.fields.forEach((f, i) => $("field").add(new Option(f.animal, i)));
$("field").onchange = e => { field = +e.target.value; e.target.blur(); draw(); };
$("help").textContent = "Keys: " + DATA.methods.map((m, i) => `${i} ${DATA.names[m]}`).join(", ")
  + "; z switches field and zoom; left and right arrows change the field.";
["field", "zoom"].forEach(v => { const b = document.createElement("button"); b.textContent = v; b.dataset.v = v;
  b.onclick = () => { view = v; draw(); }; $("views").append(b); });
DATA.methods.forEach((m, i) => { const b = document.createElement("button"); b.textContent = `${i} ${m}`; b.dataset.m = m;
  b.onclick = () => { method = m; draw(); }; $("methods").append(b); });
document.addEventListener("keydown", e => {
  if (/^[0-9]$/.test(e.key) && +e.key < DATA.methods.length) method = DATA.methods[+e.key];
  else if (e.key === "z") view = view === "zoom" ? "field" : "zoom";
  else if (e.key === "ArrowRight") field = (field + 1) % DATA.fields.length;
  else if (e.key === "ArrowLeft") field = (field + DATA.fields.length - 1) % DATA.fields.length;
  else return;
  e.preventDefault();
  draw();
});
draw();
</script></body></html>
"""


def code_version() -> str:
    """The git commit the script ran from, marked dirty if the tree had changes and untracked if it is not in it."""
    here = Path(__file__).parent
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=here)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", Path(__file__).name],
                             capture_output=True, check=False, cwd=here).returncode == 0
    version = found.stdout.strip()
    return "unknown" if not version else version if tracked else f"{version}-untracked"


def mean_or_none(values: list) -> float | None:
    """The rounded mean, or None for no values."""
    return round(float(np.mean(values)), 4) if values else None


def main() -> None:
    """Draw every field's maps, measure them on tissue, and write the PDF, the flip page and the numbers."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="H&E OME-TIFFs")
    parser.add_argument("--fields", type=Path, required=True, help="pick_fields.py output")
    parser.add_argument("--tiles", type=Path, required=True, help="nested tile predictions (A)")
    parser.add_argument("--overlap", type=Path, required=True, help="predict_field.py output (B)")
    parser.add_argument("--patches", type=Path, required=True, help="predict_field_patches.py output (D)")
    parser.add_argument("--overlap40", type=Path, required=True, help="predict_field.py output on 40 um tiles (C)")
    parser.add_argument("--pixels", type=Path, required=True, help="pixel_classifier.py output (G)")
    parser.add_argument("--lumen", type=Path, required=True, help="necrosis_lumen.py output")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--overviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")
    (args.out / "flip" / "img").mkdir(parents=True)

    chosen = json.loads(args.fields.read_text())
    export = json.loads(args.annotations.read_text())
    needed = [path for animal, field in chosen["fields"].items() for path in (
        args.slides / f"{field['slide']}.ome.tiff", args.tiles / f"{animal}_predictions.json",
        args.overlap / f"{animal}_field.json", args.patches / f"{animal}_patch_field.npz",
        args.overlap40 / f"{animal}_field.json", args.pixels / f"{animal}_pixel_field.npz",
        args.lumen / f"{animal}_lumen.npz", args.overviews / f"{field['slide']}_s0.json")]
    if missing := [str(path) for path in needed if not path.exists()]:
        raise SystemExit("missing inputs, nothing drawn:\n  " + "\n  ".join(missing))
    rows, pages = [], []
    for animal, field in chosen["fields"].items():
        info, levels = open_slide(args.slides / f"{field['slide']}.ome.tiff")
        prediction = json.loads((args.tiles / f"{animal}_predictions.json").read_text())
        level = prediction["level"]
        shape = levels[level].shape[:2]
        to_level0 = pyramid_factor(levels[0].shape[1], shape[1])
        um = info.pixel_size_um * to_level0
        half = round(chosen["size_um"] / 2 / um)
        top, left = (round(v / to_level0) - half for v in field["centre_level0_rc"])
        size = 2 * half

        rgb = np.asarray(levels[level][top:top + size, left:left + size])[..., :3]
        outline = mask_from_level(np.asarray(levels[pick_level(levels)]))
        lumen, _ = read_lumen(args.lumen / f"{animal}_lumen.npz")
        tissue = (on_level(outline, pyramid_factor(shape[1], outline.shape[1]), top, left, size)
                  & ~on_level(lumen, pyramid_factor(shape[1], lumen.shape[1]), top, left, size))

        overview = read_overview(args.overviews / f"{field['slide']}_s0.json")
        polygons = [to_level(cvat_polygons(p["points"]), overview["um_per_px"], um) - [top, left]
                    for p in export.get(f"{field['slide']}_s0.png", {"polygons": []})["polygons"]]
        drawn = (rasterise(polygons, (size, size)) if polygons else np.zeros((size, size), bool)) & tissue

        # the scores over the field and a margin, so the edge snapping sees past the field's border
        pad = 2 * (radius := round(SNAP_UM / um))
        if top < pad or left < pad or top + size + pad > shape[0] or left + size + pad > shape[1]:
            raise SystemExit(f"{animal}: the field lies within {pad} px of the slide's edge")
        inner = np.s_[pad:pad + size, pad:pad + size]
        guide = np.asarray(levels[level][top - pad:top + size + pad, left - pad:left + size + pad])[..., :3] / 255
        d_score, d_meta = map_score(args.patches / f"{animal}_patch_field.npz", "mean", top - pad, left - pad,
                                    size + 2 * pad)
        g_score, g_meta = map_score(args.pixels / f"{animal}_pixel_field.npz", "score", top, left, size)
        c_score, c_weight, c_cutoff = field_score(args.overlap40 / f"{animal}_field.json", levels, level,
                                                  (top, left), size)
        overlap40 = json.loads((args.overlap40 / f"{animal}_field.json").read_text())
        b_score, b_weight, b_cutoff = field_score(args.overlap / f"{animal}_field.json", levels, level,
                                                  (top - pad, left - pad), size + 2 * pad)
        b_score = np.where(b_weight >= 0.5, b_score, 0.0)  # no call where too few tissue tiles stand behind it
        overlap = json.loads((args.overlap / f"{animal}_field.json").read_text())
        for name, record in (("D", d_meta), ("G", g_meta)):
            if record["level"] != level:
                raise SystemExit(f"{animal}: {name}'s map is on level {record['level']}, the tiles on {level}")
        for name, record in (("B", overlap), ("C", overlap40), ("D", d_meta), ("G", g_meta)):
            if (record["field"]["centre_level0_rc"], record["slide"]) != (field["centre_level0_rc"], field["slide"]):
                raise SystemExit(f"{animal}: {name} was made for another field than --fields gives")
        calls = {
            "A": tile_calls(prediction, top, left, size),
            "B": (b_score >= b_cutoff)[inner],
            "C": (c_score >= c_cutoff) & (c_weight >= 0.5),
            "D": (d_score >= d_meta["cutoff"])[inner],
            "E": (guided_filter(guide, d_score, radius, SNAP_EPS) >= d_meta["cutoff"])[inner],
            "F": (guided_filter(guide, b_score, radius, SNAP_EPS) >= b_cutoff)[inner],
            "G": g_score >= g_meta["cutoff"],
        }
        calls = {code: call & tissue for code, call in calls.items()}

        n_tissue = max(1, int(tissue.sum()))
        area = {"drawn": float(drawn.sum() / n_tissue), **{c: float(m.sum() / n_tissue) for c, m in calls.items()}}
        scores = {c: overlap_score(drawn, m) if drawn.any() else (None, None) for c, m in calls.items()}

        # the zoom: where the methods disagree most, read at level 0 for the sharpest H&E
        zoom = round(ZOOM_UM / um)
        cy, cx, differ = disagreement_centre(calls, tissue, zoom)
        zy, zx = cy - zoom // 2, cx - zoom // 2
        z_top, z_left, z_size = (top + zy) * to_level0, (left + zx) * to_level0, zoom * to_level0
        zoom_rgb = np.asarray(levels[0][z_top:z_top + z_size, z_left:z_left + z_size])[..., :3]

        views = {
            "field": [("plain H&E", rgb, None), ("drawn polygons", rgb, drawn),
                      *[(f"{c}  {area[c]:.1%} of tissue", rgb, calls[c]) for c in calls]],
            "zoom": [("plain H&E", zoom_rgb, None), ("drawn polygons", zoom_rgb, stretched(drawn, zy, zx, zoom, z_size)),
                     *[(c, zoom_rgb, stretched(calls[c], zy, zx, zoom, z_size)) for c in calls]],
        }
        for view, panels in views.items():
            images = [(label, tinted(image, call)) for label, image, call in panels]
            placed = "where the maps disagree most" if differ else "the field centre, all maps agree"
            where = (f"centred on level-0 px {field['centre_level0_rc']}" if view == "field" else
                     f"{ZOOM_UM:.0f} um {placed}, top left at level-0 px {[z_top, z_left]}")
            title = f"{animal}, {'whole 1 mm field' if view == 'field' else f'{ZOOM_UM:.0f} um zoom'}"
            pages.append(page(title, images, f"{field['rule']}; {where}; see-through cyan on tissue, lumina out"))
            for code, (_, image) in zip(["plain", "drawn", *calls], images, strict=True):
                Image.fromarray(image).resize((FLIP_PX, FLIP_PX), Image.LANCZOS).save(
                    args.out / "flip" / "img" / f"{animal}_{view}_{code}.jpg", quality=90)

        rows.append({"animal": animal, "chosen_by": field["rule"], "area": {k: round(v, 4) for k, v in area.items()},
                     "dsc": {c: s[0] for c, s in scores.items()}, "iou": {c: s[1] for c, s in scores.items()},
                     "tissue_px": int(tissue.sum()), "zoom_level0_top_left_size": [z_top, z_left, z_size],
                     "zoom_where_maps_differ": differ,
                     "cutoffs": {"A": prediction["cutoff"], "B": b_cutoff, "C": c_cutoff, "D": d_meta["cutoff"],
                                 "E": d_meta["cutoff"], "F": b_cutoff, "G": g_meta["cutoff"]},
                     "inputs_written": {"A": prediction["written"], "B": overlap["written"], "C": overlap40["written"],
                                        "D": d_meta.get("written"), "D_code": d_meta.get("code")}})
        print(f"{animal}: drawn {area['drawn']:.1%}, " + ", ".join(f"{c} {area[c]:.1%}" for c in calls), flush=True)

    drawn_rows = [r for r in rows if r["area"]["drawn"] > 0]
    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "methods": METHODS,
        "mask": "tissue outline without lumina (necrosis_lumen.py), the same for every method and area",
        "area": "share of the field's tissue pixels called necrotic",
        "zoom_rule": (f"the {ZOOM_UM:.0f} um window holding the most tissue pixels the methods call differently, "
                      "of equal ones the nearest to those pixels' middle; the field centre where all agree"),
        "geometry": "levels mapped by their power-of-two factors, coarse pixel q centred on level pixel factor * q",
        "edge_snap": {"filter": "colour guided filter (He, Sun, Tang 2013), slideviz.analysis.edge_snap",
                      "guide": "the H&E's RGB at the tiles' level, 0 to 1", "radius_um": SNAP_UM, "eps": SNAP_EPS,
                      "applied_to": "E: D's mean score; F: B's blended score; each at its own cutoff",
                      "margin": "computed over the field plus twice the radius, then cut to the field"},
        "metric_caveat": "DSC and IoU against polygons that stop short of the necrosis edge, for information only",
        "mean_over_drawn_fields": {
            "n": len(drawn_rows),
            "dsc": {c: mean_or_none([r["dsc"][c] for r in drawn_rows]) for c in METHODS},
            "iou": {c: mean_or_none([r["iou"][c] for r in drawn_rows]) for c in METHODS},
        },
        "fields": rows,
        "inputs": {k: str(v) for k, v in vars(args).items() if k != "out"},
        "inputs_written": {"fields": chosen["written"], "fields_code": chosen["code"]},
    }
    (args.out / "necrosis_fine_report.json").write_text(json.dumps(record, indent=1) + "\n")
    pages.insert(0, summary_page(rows, record))
    pages[0].save(args.out / "necrosis_fine_report.pdf", save_all=True, append_images=pages[1:],
                  resolution=150, quality=88)
    data = {"methods": ["plain", "drawn", *METHODS], "fields": [{"animal": r["animal"]} for r in rows],
            "names": {"plain": "plain H&E", "drawn": "your polygons", **METHODS}}
    (args.out / "flip" / "index.html").write_text(FLIP_HTML.replace("__DATA__", json.dumps(data)))
    print(f"{len(rows)} fields, {len(pages)} pages, written to {args.out}")


if __name__ == "__main__":
    main()
