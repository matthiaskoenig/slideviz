"""CYP2E1-positive area and its gradient on the H&E grid, split by predicted necrosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from slideviz.analysis import dab, flatten, tissue
from slideviz.analysis.dab import (
    NO_TISSUE,
    POSITION_SCALE,
    STEPS,
    glass_colour,
    gradient_steps,
    relative_position,
    relative_readout,
)
from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.analysis.warp import load_field, nonrigid_levels
from slideviz.data.schema import Registration
from slideviz.io.reader import open_slide

WORK_UM = 1.8  # working resolution; the nearest pyramid level is used and the map is stored at it
NECROTIC_MIN_PX = 1000  # fewer necrotic pixels give no necrotic readout
DAB_SETTINGS = ("GLASS_OD", "SMOOTH_UM", "WINDOW_UM", "STEP_UM", "CELL_UM", "LOW_Q", "HIGH_Q",
                "MIDPOINT", "CONTRAST_MIN", "WINDOW_TISSUE_MIN", "STEPS", "POSITION_SCALE", "NO_TISSUE")
TISSUE_SETTINGS = ("LUMEN_OPEN_UM", "LUMEN_MIN_UM2", "RIM_UM")
FLATTEN_DIR = "flatten"  # flattening fields beside the slides, where the viewer finds them

COLUMNS = 4
PANEL_PX = 300
CAPTION_PX = 22
SHEET_STRIDE = 4  # the sheet reads every 4th map pixel
NEGATIVE, POSITIVE, OUTSIDE = (236, 236, 244), (139, 69, 19), (154, 154, 154)
# one colour per step from negative to positive, then outside, indexed by gradient_steps
STEP_COLOURS = np.array([*np.linspace(NEGATIVE, POSITIVE, STEPS), OUTSIDE]).round().astype(np.uint8)


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


def percent(part: np.ndarray, whole: np.ndarray) -> float | None:
    """Share of `whole` that is also `part`, in percent."""
    return round(100 * float((part & whole).sum() / whole.sum()), 1) if whole.any() else None


def counts(steps: np.ndarray, where: np.ndarray) -> list[int]:
    """Pixels in each gradient step within `where`."""
    return np.bincount(steps[where], minlength=STEPS + 1)[:STEPS].tolist()


def shares(counted: list[int]) -> list[float] | None:
    """Step counts as percent of their sum."""
    total = sum(counted)
    return [round(100 * c / total, 1) for c in counted] if total else None


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    """Overlap of two masks, twice their intersection over their summed sizes."""
    size = int(a.sum() + b.sum())
    return round(2 * float((a & b).sum()) / size, 3) if size else None


def readout(rgb: np.ndarray, glass: np.ndarray, outline: np.ndarray, um_per_px: float,
            necrosis: tuple[np.ndarray, np.ndarray] | None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Numbers, positive map and stored position for one RGB level, split by necrosis when given."""
    positive, smoothed, levels = relative_readout(rgb, glass, outline, um_per_px)
    in_tissue = positive != 2
    position = relative_position(smoothed, in_tissue, levels)
    del smoothed
    steps = gradient_steps(position)
    hit = positive == 1
    pixels = {"tissue": int(in_tissue.sum()), "positive": int(hit.sum()), "steps": counts(steps, in_tissue)}
    numbers = {
        "tissue_mm2": round(pixels["tissue"] * (um_per_px / 1000) ** 2, 2),
        "positive_pct": percent(hit, in_tissue),
        "steps_pct": shares(pixels["steps"]),
        "flat_windows": levels.flat_fraction,
    }
    if necrosis is not None:
        necrotic, covered = necrosis
        surviving = in_tissue & covered & ~necrotic
        dead = in_tissue & covered & necrotic
        enough = dead.sum() >= NECROTIC_MIN_PX
        pixels |= {
            "covered": int((in_tissue & covered).sum()),
            "surviving": int(surviving.sum()),
            "necrotic": int(dead.sum()),
            "positive_surviving": int((hit & surviving).sum()),
            "positive_necrotic": int((hit & dead).sum()),
            "steps_surviving": counts(steps, surviving),
            "steps_necrotic": counts(steps, dead),
        }
        numbers |= {
            "necrotic_pct": percent(necrotic, in_tissue & covered),
            "positive_surviving_pct": percent(hit, surviving),
            "positive_necrotic_pct": percent(hit, dead) if enough else None,
            "steps_surviving_pct": shares(pixels["steps_surviving"]),
            "steps_necrotic_pct": shares(pixels["steps_necrotic"]) if enough else None,
            "dice_positive_necrotic": dice(hit & covered, dead) if enough else None,
        }
    return numbers | {"pixels": pixels}, positive, position


def measure(job: tuple) -> dict:
    """One CYP2E1 slide: readout and gradient on its H&E partner's grid, raw and optionally flattened."""
    cyp_stem, slides, prediction, out, flattened, provenance = job
    side = sidecar(slides, cyp_stem)
    registration = Registration.model_validate(side["registration"])
    field = load_field(slides, registration)
    if field is None:
        raise RuntimeError(f"{cyp_stem}: no valid non-rigid field")

    he_info, he = open_slide(slides / f"{registration.reference}.ome.tiff")
    cyp_info, cyp = open_slide(slides / f"{cyp_stem}.ome.tiff")
    he_shapes = [level.shape[:2] for level in he]
    warped = nonrigid_levels(cyp, he_shapes, registration, field, he_info.pixel_size_um)
    index = level_near(he, he_info.pixel_size_um, WORK_UM)
    um_per_px = he_info.pixel_size_um * he[0].shape[1] / he[index].shape[1]

    coarse = np.asarray(warped[pick_level(warped)])
    outline = mask_from_level(coarse)
    glass = glass_colour(coarse, outline)
    record = json.loads(prediction.read_text()) if prediction is not None else None
    necrosis = necrosis_on(record, he_shapes[index], he, index) if record is not None else None

    numbers, positive, position = readout(np.asarray(warped[index]), glass, outline, um_per_px, necrosis)
    row = {
        "he": registration.reference,
        "dose_mg_per_kg": side.get("dose_mg_per_kg"),
        "serial_block": side.get("serial_block"),
        "registration": [registration.method, registration.nonrigid.method],
        "predictions": prediction.name if prediction is not None else None,
        "necrosis_cutoff": record["cutoff"] if record is not None else None,
        **numbers,
    }

    if flattened:
        path = slides / FLATTEN_DIR / f"{cyp_stem}_flatten.npz"
        # flattened in the slide's own frame, then warped, as the viewer draws it
        evened = nonrigid_levels(flatten.flatten_levels(cyp, flatten.load(path), cyp_info.pixel_size_um),
                                 he_shapes, registration, field, he_info.pixel_size_um)
        flat_numbers, flat_positive, flat_position = readout(
            np.asarray(evened[index]), glass, outline, um_per_px, necrosis)
        both = (positive != 2) & (flat_positive != 2)
        row["flattened"] = {
            "fields": path.name,
            "fields_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "changed_call_pct": percent(positive != flat_positive, both),
            "changed_step_pct": percent(gradient_steps(position) != gradient_steps(flat_position), both),
            **flat_numbers,
        }

    row["map"] = f"{cyp_stem}_readout.npz"
    meta = {
        "slide": cyp_stem,
        "reference": registration.reference,
        "registration": row["registration"],
        "predictions": row["predictions"],
        "map": f"position between the window's low (0) and high ({POSITION_SCALE}) DAB, {NO_TISSUE} outside the tissue",
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        **provenance,
    }
    np.savez_compressed(out / row["map"], position=position, um_per_px=um_per_px,
                        reference=registration.reference, meta=json.dumps(meta))
    flat_row = row.get("flattened")
    print(f"{cyp_stem}: {row['positive_pct']}% positive, surviving {row.get('positive_surviving_pct')}%, "
          f"necrotic {row.get('necrotic_pct')}%"
          + (f", flattened {flat_row['positive_pct']}%" if flat_row else ""), flush=True)
    return {"stem": cyp_stem, "row": row, "um_per_px": um_per_px,
            "raw": np.asarray(warped[min(index + 2, len(warped) - 1)]),
            "steps": gradient_steps(position[::SHEET_STRIDE, ::SHEET_STRIDE])}


def panel(image: np.ndarray) -> Image.Image:
    """An image scaled to fit one sheet panel."""
    picture = Image.fromarray(image)
    picture.thumbnail((PANEL_PX, PANEL_PX))
    return picture


def write_sheet(results: list[dict], path: Path) -> None:
    """One PNG: raw CYP2E1 on the H&E grid beside its gradient map, for every slide."""
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
        sheet.paste(panel(STEP_COLOURS[result["steps"]]), (x + PANEL_PX, y + CAPTION_PX))
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
    parser.add_argument("--flattened", action="store_true",
                        help=f"also measure on flattened pixels, fields from <slides>/{FLATTEN_DIR}/")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    predictions = {}
    if args.predictions:
        for path in sorted(args.predictions.glob("*_predictions.json")):
            record = json.loads(path.read_text())
            if "cutoff" not in record:
                raise SystemExit(f"{path} has no nested cutoff; use the nested prediction files")
            predictions[record["slide"]] = path

    code = code_version()
    settings = ({name: getattr(dab, name) for name in DAB_SETTINGS}
                | {name: getattr(tissue, name) for name in TISSUE_SETTINGS})
    jobs = []
    for path in sorted(args.slides.glob("*_cyp2e1.ome.tiff")):
        stem = path.name.removesuffix(".ome.tiff")
        if args.blocks and not any(block in stem for block in args.blocks):
            continue
        reference = sidecar(args.slides, stem)["registration"]["reference"]
        jobs.append((stem, args.slides, predictions.get(reference), args.out, args.flattened,
                     {"code": code, "settings": settings}))
    if not jobs:
        raise SystemExit(f"no CYP2E1 slides in {args.slides}")
    if args.flattened:
        missing = [job[0] for job in jobs if not (args.slides / FLATTEN_DIR / f"{job[0]}_flatten.npz").exists()]
        if missing:
            raise SystemExit(f"no flattening fields in {args.slides / FLATTEN_DIR} for {', '.join(missing)}")

    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(measure, jobs)

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code,
        "slides": str(args.slides),
        "predictions": str(args.predictions) if args.predictions else None,
        "flattened": str(args.slides / FLATTEN_DIR) if args.flattened else None,
        "method": "relative DAB readout and its gradient on the H&E grid, rigid plus non-rigid registration",
        "stain_vectors": "Ruifrok and Johnston 2001, skimage.color.hed_from_rgb",
        "work_um_per_px": round(results[0]["um_per_px"], 4),
        "map": f"position between the window's low (0) and high ({POSITION_SCALE}) DAB per working pixel, "
               f"uint8, {NO_TISSUE} outside the tissue; steps from dab.gradient_steps",
        "settings": settings,
        "slides_measured": {r["stem"]: r["row"] for r in results},
    }
    (args.out / "cyp2e1_readout.json").write_text(json.dumps(record, indent=1))
    if args.sheet:
        write_sheet(results, args.sheet)
    print(f"{len(results)} slides, written to {args.out}")


if __name__ == "__main__":
    main()
