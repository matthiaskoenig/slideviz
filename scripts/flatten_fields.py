"""Learn shared DAB rim decay, then write flattening fields for slides.

Example:
    uv run python scripts/flatten_fields.py --slides /data/michelle/slides \
        --predictions /data/michelle/slides/predictions --out /data/michelle/flatten_run --workers 10
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from slideviz.analysis import flatten
from slideviz.analysis.dab import glass_colour
from slideviz.analysis.tiling import pick_level_um
from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.data.provenance import code_version
from slideviz.data.registration import read_sidecar
from slideviz.io.reader import open_slide

METHOD = {
    "he": "per channel: 1 mm trend plus up to 4 feathered paler or darker regions",
    "dab": "counterstain: 1 mm trend; brown (DAB and middle channel together): 2 mm trend plus "
           "edge rim, strength x exp(-depth / shared decay) per outline sector on the 25th percentile, "
           "70 % removed",
}


def view_of(path: Path) -> dict:
    """The level nearest flatten.VIEW_UM, its pixel size, tissue mask and glass colour."""
    info, levels = open_slide(path)
    index = pick_level_um([level.shape for level in levels], info.pixel_size_um, flatten.VIEW_UM)
    view = np.asarray(levels[index])[..., :3]
    view_um = info.pixel_size_um * levels[0].shape[1] / levels[index].shape[1]
    # the tissue mask is tuned for the ~600 px level, so it is made there and resized
    mask = mask_from_level(np.asarray(levels[pick_level(levels)])[..., :3])
    tissue = resize(mask, view.shape[:2], order=0, preserve_range=True, anti_aliasing=False).astype(bool)
    return {"view": view, "view_um": view_um, "tissue": tissue, "glass": glass_colour(view, tissue),
            "level0_um": info.pixel_size_um, "levels": levels}


def necrosis_view(record: dict, levels: list, level0_um: float, shape, view_um: float) -> np.ndarray:
    """Tiles at or above the nested cutoff, on a view grid of the predicted slide."""
    level_um = level0_um * levels[0].shape[1] / levels[record["level"]].shape[1]
    scale = level_um / view_um
    size = max(1, math.ceil(record["size_px"] * scale))
    mask = np.zeros(shape, bool)
    for tile in record["tiles"]:
        if tile["predicted"] >= record["cutoff"]:
            r, c = int(tile["y"] * scale), int(tile["x"] * scale)
            mask[r:r + size, c:c + size] = True
    return mask


def carry(mask: np.ndarray, mask_um: float, ref_level0_um: float, matrix: list, shape, view_um: float,
          level0_um: float) -> np.ndarray:
    """A mask on the reference slide's view, sampled onto the moving slide's view by the rigid matrix."""
    m = np.asarray(matrix, float)  # moving level-0 x/y onto reference level-0 x/y
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    x, y = (cols + 0.5) * view_um / level0_um, (rows + 0.5) * view_um / level0_um
    px, py = m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]
    to_mask = ref_level0_um / mask_um
    return map_coordinates(mask.astype(float), [py * to_mask - 0.5, px * to_mask - 0.5],
                           order=0, cval=0) > 0.5


def grid_for(job: tuple) -> tuple[str, str, flatten.Grid, dict]:
    """One slide's fitting grid, its stain kind and the inputs it used."""
    stem, slides, predictions = job
    car = read_sidecar(slides, stem)
    kind = "dab" if car["stain"] in flatten.DAB_STAINS else "he"
    v = view_of(slides / f"{stem}.ome.tiff")
    inputs = {"slide": f"{stem}.ome.tiff", "view_um": round(v["view_um"], 4), "prediction": None}
    reference = car["registration"]["reference"] if kind == "dab" else stem
    necrotic = np.zeros(v["tissue"].shape, bool)
    if reference in predictions:
        record = json.loads(predictions[reference].read_text())
        inputs["prediction"] = predictions[reference].name
        if kind == "he":
            necrotic = necrosis_view(record, v["levels"], v["level0_um"], necrotic.shape, v["view_um"])
        else:
            ref = view_of(slides / f"{reference}.ome.tiff")
            ref_dead = necrosis_view(record, ref["levels"], ref["level0_um"], ref["tissue"].shape,
                                     ref["view_um"])
            necrotic = carry(ref_dead, ref["view_um"], ref["level0_um"], car["registration"]["matrix"],
                             necrotic.shape, v["view_um"], v["level0_um"])
            inputs["registration_reference"] = reference
    grid = flatten.prepare(v["view"], v["view_um"], v["tissue"], necrotic, v["glass"])
    return stem, kind, grid, inputs


def fields_for(job: tuple) -> dict:
    """Fit and write one slide's fields; the row for the run record."""
    stem, kind, grid, inputs, decay, typical, out, code = job
    row = {"stain_kind": kind, **inputs,
           "trusted_share": round(float(grid.trust.sum() / grid.tissue.sum()), 3)}
    if kind == "he":
        fields = flatten.he_fields(grid)
    else:
        fields, strength = flatten.dab_fields(grid, decay, typical)
        row["rim_strength"] = [round(float(np.exp(s)), 3) for s in strength]
    row["field_range"] = {ch: [round(float(np.exp(f[grid.tissue].min())), 3),
                               round(float(np.exp(f[grid.tissue].max())), 3)] for ch, f in fields.items()}
    meta = {"slide": stem, "method": METHOD[kind], "code": code,
            "written": datetime.now(UTC).isoformat(timespec="seconds"), "inputs": inputs,
            "settings": settings()}
    if kind == "dab":
        meta |= {"rim_decay_um": round(decay, 1), "typical_rim": round(float(np.exp(typical)), 3),
                 "rim_strength": row["rim_strength"]}
    flatten.save(out / f"{stem}_flatten.npz",
                 flatten.Flattening(fields, grid.tissue, grid.grid_um, grid.glass, meta))
    print(f"{stem}: {kind}, trusted {row['trusted_share']:.0%}", flush=True)
    return row


def settings() -> dict:
    """Every setting of the flatten module, by name."""
    return {name: getattr(flatten, name) for name in dir(flatten)
            if name.isupper() and isinstance(getattr(flatten, name), int | float | tuple | str)}


def main() -> None:
    """Learn the shared rim from all DAB slides, then write fields for the chosen slides."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="OME-TIFFs with their sidecars")
    parser.add_argument("--predictions", type=Path, required=True, help="nested necrosis predictions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", nargs="*", help="write fields only for stems containing one of these")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for path in sorted(args.predictions.glob("*_predictions.json")):
        record = json.loads(path.read_text())
        if "cutoff" not in record:
            raise SystemExit(f"{path} has no nested cutoff; use the nested prediction files")
        predictions[record["slide"]] = path
    stems = sorted(p.name.removesuffix(".ome.tiff") for p in args.slides.glob("*.ome.tiff"))
    if not stems:
        raise SystemExit(f"no slides in {args.slides}")
    wanted = [s for s in stems if not args.blocks or any(b in s for b in args.blocks)]
    dab = [s for s in stems if read_sidecar(args.slides, s)["stain"] in flatten.DAB_STAINS]
    jobs = sorted(set(wanted) | set(dab))

    with Pool(max(1, min(args.workers, len(jobs)))) as pool:
        grids = {stem: (kind, grid, inputs)
                 for stem, kind, grid, inputs in pool.map(grid_for, [(s, args.slides, predictions) for s in jobs])}

    decay, typical, informing = None, None, []
    if dab:
        profiles = {s: flatten.deep_profile(grids[s][1]) for s in dab}
        informing = [s for s, p in profiles.items() if p is not None]
        decay = flatten.rim_decay([profiles[s] for s in informing])
        strengths = [flatten.brown_parts(grids[s][1], decay)[2] for s in informing]
        typical = float(np.median(np.concatenate(strengths)))
        print(f"rim decay {decay:.0f} um, typical rim x{np.exp(typical):.2f}, from {len(informing)} slides",
              flush=True)

    code = code_version(Path(__file__))
    with Pool(max(1, min(args.workers, len(wanted)))) as pool:
        rows = pool.map(fields_for, [(s, *grids[s], decay, typical, args.out, code) for s in wanted])

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code,
        "slides": str(args.slides),
        "predictions": str(args.predictions),
        "method": METHOD,
        "stain_vectors": "Ruifrok and Johnston 2001, skimage.color.hed_from_rgb",
        "rim_decay_um": round(decay, 1) if decay else None,
        "typical_rim": round(float(np.exp(typical)), 3) if typical is not None else None,
        "rim_learned_from": informing,
        "settings": settings(),
        "fields": dict(zip(wanted, rows, strict=True)),
    }
    (args.out / "flatten.json").write_text(json.dumps(record, indent=1))
    print(f"{len(rows)} slides, written to {args.out}")


if __name__ == "__main__":
    main()
