"""CYP2E1-positive area on the H&E grid, split by predicted necrosis."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from slideviz.analysis import dab, tissue
from slideviz.analysis.dab import glass_colour, relative_readout
from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.analysis.warp import load_field, nonrigid_levels
from slideviz.data.schema import Registration
from slideviz.io.reader import open_slide

WORK_UM = 1.8  # working resolution; the nearest pyramid level is used
MAP_UM = 7.0  # stored map cell, finer than one zone and small on disk
NECROTIC_MIN_PX = 1000  # fewer necrotic pixels give no necrotic readout
DAB_SETTINGS = ("GLASS_OD", "SMOOTH_UM", "WINDOW_UM", "STEP_UM", "CELL_UM", "LOW_Q", "HIGH_Q",
                "MIDPOINT", "CONTRAST_MIN", "WINDOW_TISSUE_MIN")
TISSUE_SETTINGS = ("LUMEN_OPEN_UM", "LUMEN_MIN_UM2", "RIM_UM")

COLUMNS = 4
PANEL_PX = 300
CAPTION_PX = 22
POSITIVE, NEGATIVE, OUTSIDE = (139, 69, 19), (236, 236, 244), (154, 154, 154)


def sidecar(slides: Path, stem: str) -> dict:
    """A slide's sidecar, named <stem>.json or <stem>.ome.json."""
    for name in (f"{stem}.json", f"{stem}.ome.json"):
        if (slides / name).exists():
            return json.loads((slides / name).read_text())
    raise FileNotFoundError(f"no sidecar for {stem} in {slides}")


def level_near(levels: list, pixel_size_um: float, target_um: float) -> int:
    """Index of the level whose pixel size is closest to target_um."""
    sizes = [pixel_size_um * levels[0].shape[1] / level.shape[1] for level in levels]
    return min(range(len(levels)), key=lambda i: abs(sizes[i] - target_um))


def necrosis_on(record: dict, shape: tuple[int, int], he: list, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicted necrosis above the nested cutoff, and tile coverage, on H&E level `index`."""
    size, level = record["size_px"], record["level"]
    tiles = record["tiles"]
    necrotic = np.zeros((max(t["row"] for t in tiles) + 1, max(t["col"] for t in tiles) + 1), bool)
    covered = np.zeros_like(necrotic)
    for t in tiles:
        necrotic[t["row"], t["col"]] = t["predicted"] >= record["cutoff"]
        covered[t["row"], t["col"]] = True
    # level `index` pixels to tile indices, through the prediction's own level
    scale = he[level].shape[1] / he[index].shape[1] / size
    rows = np.minimum((np.arange(shape[0]) * scale).astype(int), necrotic.shape[0] - 1)
    cols = np.minimum((np.arange(shape[1]) * scale).astype(int), necrotic.shape[1] - 1)
    inside = ((np.arange(shape[0]) * scale) < necrotic.shape[0])[:, None] & (
        (np.arange(shape[1]) * scale) < necrotic.shape[1])[None, :]
    return necrotic[np.ix_(rows, cols)] & inside, covered[np.ix_(rows, cols)] & inside


def share_map(positive: np.ndarray, block: int) -> np.ndarray:
    """Positive share of the tissue in each block x block cell, NaN where there is none."""
    h, w = (positive.shape[0] // block) * block, (positive.shape[1] // block) * block
    cells = positive[:h, :w].reshape(h // block, block, w // block, block)
    in_tissue = (cells != 2).sum(axis=(1, 3))
    hits = (cells == 1).sum(axis=(1, 3))
    return np.where(in_tissue > 0, hits / np.maximum(in_tissue, 1), np.nan).astype(np.float32)


def percent(part: np.ndarray, whole: np.ndarray) -> float | None:
    """Share of `whole` that is also `part`, in percent."""
    return round(100 * float((part & whole).sum() / whole.sum()), 1) if whole.any() else None


def measure(job: tuple) -> dict:
    """One CYP2E1 slide: readout on its H&E partner's grid, stored map and numbers."""
    cyp_stem, slides, prediction, out = job
    side = sidecar(slides, cyp_stem)
    registration = Registration.model_validate(side["registration"])
    field = load_field(slides, registration)
    if field is None:
        raise RuntimeError(f"{cyp_stem}: no valid non-rigid field")

    he_info, he = open_slide(slides / f"{registration.reference}.ome.tiff")
    _, cyp = open_slide(slides / f"{cyp_stem}.ome.tiff")
    warped = nonrigid_levels(cyp, [level.shape[:2] for level in he], registration, field,
                             he_info.pixel_size_um)
    index = level_near(he, he_info.pixel_size_um, WORK_UM)
    um_per_px = he_info.pixel_size_um * he[0].shape[1] / he[index].shape[1]

    coarse = np.asarray(warped[pick_level(warped)])
    outline = mask_from_level(coarse)
    positive, _, levels = relative_readout(
        np.asarray(warped[index]), glass_colour(coarse, outline), outline, um_per_px)
    in_tissue = positive != 2
    hit = positive == 1

    row = {
        "he": registration.reference,
        "dose_mg_per_kg": side.get("dose_mg_per_kg"),
        "serial_block": side.get("serial_block"),
        "registration": [registration.method, registration.nonrigid.method],
        "tissue_mm2": round(float(in_tissue.sum()) * (um_per_px / 1000) ** 2, 2),
        "positive_pct": percent(hit, in_tissue),
        "flat_windows": levels.flat_fraction,
        "predictions": None,
    }
    if prediction is not None:
        record = json.loads(prediction.read_text())
        necrotic, covered = necrosis_on(record, positive.shape, he, index)
        dead = in_tissue & covered & necrotic
        row |= {
            "predictions": prediction.name,
            "necrosis_cutoff": record["cutoff"],
            "necrotic_pct": percent(necrotic, in_tissue & covered),
            "positive_surviving_pct": percent(hit, in_tissue & covered & ~necrotic),
            "positive_necrotic_pct": percent(hit, dead) if dead.sum() >= NECROTIC_MIN_PX else None,
        }

    block = max(1, round(MAP_UM / um_per_px))
    shares = share_map(positive, block)
    np.savez_compressed(out / f"{cyp_stem}_readout.npz", positive_share=shares,
                        um_per_px=um_per_px * block, reference=registration.reference)
    row["map"] = f"{cyp_stem}_readout.npz"
    print(f"{cyp_stem}: {row['positive_pct']}% positive, surviving "
          f"{row.get('positive_surviving_pct')}%, necrotic {row.get('necrotic_pct')}%", flush=True)
    return {"stem": cyp_stem, "row": row, "um_per_px": um_per_px,
            "raw": np.asarray(warped[min(index + 2, len(warped) - 1)]), "shares": shares}


def panel(image: np.ndarray) -> Image.Image:
    """An image scaled to fit one sheet panel."""
    picture = Image.fromarray(image)
    picture.thumbnail((PANEL_PX, PANEL_PX))
    return picture


def coloured(shares: np.ndarray) -> np.ndarray:
    """A share map as positive, negative and outside colours, split at one half."""
    out = np.empty((*shares.shape, 3), np.uint8)
    out[:] = OUTSIDE
    out[shares >= 0.5] = POSITIVE
    out[shares < 0.5] = NEGATIVE
    return out


def write_sheet(results: list[dict], path: Path) -> None:
    """One PNG: raw CYP2E1 on the H&E grid beside its positive map, for every slide."""
    rows = math.ceil(len(results) / COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * 2 * PANEL_PX, rows * (PANEL_PX + CAPTION_PX)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, result in enumerate(results):
        x = (i % COLUMNS) * 2 * PANEL_PX
        y = (i // COLUMNS) * (PANEL_PX + CAPTION_PX)
        row = result["row"]
        surviving = row.get("positive_surviving_pct")
        draw.text((x + 4, y + 4), f"{result['stem']}: {row['positive_pct']}% positive"
                  + (f", {surviving}% of surviving" if surviving is not None else ""), fill="black")
        sheet.paste(panel(result["raw"]), (x, y + CAPTION_PX))
        sheet.paste(panel(coloured(result["shares"])), (x + PANEL_PX, y + CAPTION_PX))
    sheet.save(path)


def code_version() -> str:
    """The git commit the script ran from, marked dirty if the tree had changes."""
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=Path(__file__).parent)
    return found.stdout.strip() or "unknown"


def main() -> None:
    """Measure every CYP2E1 slide in --slides and write maps, numbers and settings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="nested necrosis predictions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", nargs="*", help="only stems containing one of these")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sheet", type=Path, help="contact sheet PNG")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    predictions = {}
    if args.predictions:
        for path in sorted(args.predictions.glob("*_predictions.json")):
            record = json.loads(path.read_text())
            if "cutoff" not in record:
                raise SystemExit(f"{path} has no nested cutoff; use the nested prediction files")
            predictions[record["slide"]] = path

    jobs = []
    for path in sorted(args.slides.glob("*_cyp2e1.ome.tiff")):
        stem = path.name.removesuffix(".ome.tiff")
        if args.blocks and not any(block in stem for block in args.blocks):
            continue
        reference = sidecar(args.slides, stem)["registration"]["reference"]
        jobs.append((stem, args.slides, predictions.get(reference), args.out))
    if not jobs:
        raise SystemExit(f"no CYP2E1 slides in {args.slides}")

    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(measure, jobs)

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "slides": str(args.slides),
        "predictions": str(args.predictions) if args.predictions else None,
        "method": "relative DAB readout on the H&E grid, rigid plus non-rigid registration",
        "stain_vectors": "Ruifrok and Johnston 2001, skimage.color.hed_from_rgb",
        "work_um_per_px": round(results[0]["um_per_px"], 4),
        "map_um_per_px": MAP_UM,
        "settings": {name: getattr(dab, name) for name in DAB_SETTINGS}
        | {name: getattr(tissue, name) for name in TISSUE_SETTINGS},
        "slides_measured": {r["stem"]: r["row"] for r in results},
    }
    (args.out / "cyp2e1_readout.json").write_text(json.dumps(record, indent=1))
    if args.sheet:
        write_sheet(results, args.sheet)
    print(f"{len(results)} slides, written to {args.out}")


if __name__ == "__main__":
    main()
