"""CYP2E1 landscape and lobule borders on the H&E grid, one map per CYP2E1 slide, for the viewer.

    uv run python scripts/zonation_layers.py --slides /data/michelle/slides --out <new dir> --sheet <png>

The CYP2E1 slide is warped onto its H&E partner (rigid plus non-rigid), as cyp2e1_readout.py does, and measured there:
DAB optical density smoothed over cells, then the landscape of slideviz.analysis.zonation. Lobules are the watershed
basins of the landscape's peaks, grown over lumina; a border pixel has another lobule below or to its right. Writes
<stem>_lobules.npz per slide and the run record.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import expand_labels, watershed
from skimage.transform import resize

from slideviz.analysis import dab, tissue, zonation
from slideviz.analysis.tiling import pick_level_um
from slideviz.analysis.tissue import mask_from_level, pick_level, working_mask
from slideviz.analysis.warp import load_field, nonrigid_levels
from slideviz.data.provenance import code_version
from slideviz.data.registration import read_sidecar
from slideviz.data.schema import Registration
from slideviz.io.reader import open_slide

HOLE_UM = 350.0  # basins grow this far over lumina and gaps, so borders run on between lobules
LOW_PCT, HIGH_PCT = 1, 99  # percentiles of the landscape over profile tissue: the display range
RECORD = "zonation_layers.json"
SUFFIX = "_lobules.npz"
ZONATION_SETTINGS = ("LANDSCAPE_UM", "BLOCK_TISSUE_MIN", "SIGMA_UM", "PEAK_GAP_UM", "EDGE_UM")
DAB_SETTINGS = ("WORK_UM", "GLASS_OD", "SMOOTH_UM", "CELL_UM")
TISSUE_SETTINGS = ("LUMEN_OPEN_UM", "LUMEN_MIN_UM2", "RIM_UM")

COLUMNS = 4
PANEL_PX = 420
CAPTION_PX = 22
PALE, BROWN, LINE = (255, 250, 220), (120, 50, 5), (0, 0, 0)


def borders(lobules: np.ndarray) -> np.ndarray:
    """Pixels whose lower or right neighbour belongs to another lobule; 0 is no lobule."""
    out = np.zeros(lobules.shape, bool)
    a, b = lobules[:-1], lobules[1:]
    out[:-1] |= (a != b) & (a > 0) & (b > 0)
    a, b = lobules[:, :-1], lobules[:, 1:]
    out[:, :-1] |= (a != b) & (a > 0) & (b > 0)
    return out


def measure(job: tuple) -> dict:
    """One CYP2E1 slide: landscape, lobule borders, peaks and valleys on its H&E partner's grid."""
    stem, slides, out, provenance = job
    registration = Registration.model_validate(read_sidecar(slides, stem)["registration"])
    field = load_field(slides, registration)
    if field is None:
        raise RuntimeError(f"{stem}: no valid non-rigid field")

    he_info, he = open_slide(slides / f"{registration.reference}.ome.tiff")
    _, cyp = open_slide(slides / f"{stem}.ome.tiff")
    he_shapes = [level.shape[:2] for level in he]
    warped = nonrigid_levels(cyp, he_shapes, registration, field, he_info.pixel_size_um)
    index = pick_level_um([level.shape for level in he], he_info.pixel_size_um, dab.WORK_UM)
    um = he_info.pixel_size_um * he[0].shape[1] / he[index].shape[1]

    coarse = np.asarray(warped[pick_level(warped)])
    outline = mask_from_level(coarse)
    od = dab.stain_od(np.asarray(warped[index]), dab.glass_colour(coarse, outline))
    cells = working_mask(outline, dab.glass_like(od), um)
    inside = resize(outline, cells.shape, order=0).astype(bool)
    land = zonation.landscape(dab.smooth(od[..., dab.DAB], cells, um), cells, inside, um)
    del od

    markers = np.zeros(land.field.shape, np.int32)
    markers[tuple(land.peaks.T)] = np.arange(1, len(land.peaks) + 1)
    basins = watershed(-land.field, markers, mask=land.solid)
    within = zonation.block_means(inside, np.ones(inside.shape, bool), land.block)[0] >= 0.5
    lobules = np.where(within, expand_labels(basins, round(HOLE_UM / land.um_per_px)), 0)
    border = borders(lobules)

    scored = land.usable if land.usable.any() else land.solid
    low, high = (float(v) for v in np.percentile(land.field[scored], (LOW_PCT, HIGH_PCT)))
    offset = (land.block - 1) / 2 * um
    meta = {
        "slide": stem,
        "reference": registration.reference,
        "registration": [registration.method, registration.nonrigid.method],
        "frame": "H&E level 0 um; pixel (r, c) is centred at offset_um + (r, c) * um_per_px",
        "landscape": "DAB optical density blurred over tissue (zonation.landscape), NaN outside solid tissue",
        "border": "1 where the lobule below or to the right is another one",
        "low_high": f"p{LOW_PCT} and p{HIGH_PCT} of the landscape over profile tissue",
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        **provenance,
    }
    name = f"{stem}{SUFFIX}"
    np.savez_compressed(
        out / name, landscape=np.where(land.solid, land.field, np.nan).astype(np.float32),
        border=border.astype(np.uint8), peaks=land.peaks.astype(np.int32), valleys=land.valleys.astype(np.int32),
        um_per_px=land.um_per_px, offset_um=offset, low=low, high=high, reference=registration.reference,
        nonrigid_sha256=registration.nonrigid.sha256, meta=json.dumps(meta))
    print(f"{stem}: {len(land.peaks)} lobules, {len(land.valleys)} valleys, grid {land.field.shape}", flush=True)

    shade = np.clip((land.field - low) / (high - low), 0, 1)[..., None]
    picture = (np.array(PALE) + shade * (np.array(BROWN) - np.array(PALE))).round().astype(np.uint8)
    picture[~land.solid] = 255
    picture[border] = LINE
    thumb = np.asarray(warped[min(index + 2, len(warped) - 1)]).copy()
    lines = np.asarray(Image.fromarray(border).resize(thumb.shape[1::-1], Image.Resampling.NEAREST))
    thumb[lines] = LINE
    return {"stem": stem, "raw": thumb, "picture": picture,
            "row": {"he": registration.reference, "map": name, "lobules": len(land.peaks),
                    "valleys": len(land.valleys), "grid": list(land.field.shape),
                    "um_per_px": round(land.um_per_px, 4), "low": round(low, 5), "high": round(high, 5)}}


def panel(image: np.ndarray) -> Image.Image:
    """An image scaled to fit one sheet panel."""
    picture = Image.fromarray(image)
    picture.thumbnail((PANEL_PX, PANEL_PX))
    return picture


def write_sheet(results: list[dict], path: Path) -> None:
    """One PNG: the CYP2E1 on the H&E grid with the borders beside the landscape with the borders, per slide."""
    rows = math.ceil(len(results) / COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * 2 * PANEL_PX, rows * (PANEL_PX + CAPTION_PX)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, result in enumerate(results):
        x = (i % COLUMNS) * 2 * PANEL_PX
        y = (i // COLUMNS) * (PANEL_PX + CAPTION_PX)
        draw.text((x + 4, y + 4), f"{result['stem']}: {result['row']['lobules']} lobules", fill="black")
        sheet.paste(panel(result["raw"]), (x, y + CAPTION_PX))
        sheet.paste(panel(result["picture"]), (x + PANEL_PX, y + CAPTION_PX))
    sheet.save(path)


def main() -> None:
    """Measure every CYP2E1 slide in --slides and write the maps, the record and the sheet."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", nargs="*", help="only stems containing one of these")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sheet", type=Path, help="contact sheet PNG")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    code = code_version(Path(__file__))
    settings = ({name: getattr(zonation, name) for name in ZONATION_SETTINGS}
                | {name: getattr(dab, name) for name in DAB_SETTINGS}
                | {name: getattr(tissue, name) for name in TISSUE_SETTINGS}
                | {"HOLE_UM": HOLE_UM, "LOW_PCT": LOW_PCT, "HIGH_PCT": HIGH_PCT})
    jobs = []
    for path in sorted(args.slides.glob("*_cyp2e1.ome.tiff")):
        stem = path.name.removesuffix(".ome.tiff")
        if args.blocks and not any(block in stem for block in args.blocks):
            continue
        jobs.append((stem, args.slides, args.out, {"code": code, "settings": settings}))
    if not jobs:
        raise SystemExit(f"no CYP2E1 slides in {args.slides}")

    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(measure, jobs)

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code,
        "slides": str(args.slides),
        "method": "CYP2E1 landscape and watershed lobule borders on the H&E grid, rigid plus non-rigid registration",
        "map": "landscape float32 (DAB OD, NaN outside solid tissue), border uint8, peaks and valleys as landscape "
               "(row, col); pixel (r, c) is centred at offset_um + (r, c) * um_per_px in H&E level 0 um",
        "settings": settings,
        "slides_measured": {r["stem"]: r["row"] for r in results},
    }
    (args.out / RECORD).write_text(json.dumps(record, indent=1))
    if args.sheet:
        write_sheet(results, args.sheet)
    print(f"{len(results)} slides, written to {args.out}")


if __name__ == "__main__":
    main()
