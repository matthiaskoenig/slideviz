"""Stain profiles along the lobule position, 0 periportal to 1 pericentral, per slide and as a group mean with SD and SE.

    uv run python scripts/zonation_profiles.py --slides /data/michelle/slides --stain cyp2e1 \
        --blocks 000mg_ --out <new dir> --workers 3

Each slide is measured in its own frame on the working grid (dab.WORK_UM): DAB optical density smoothed over cells
(dab.smooth), and the position p of slideviz.analysis.zonation, from points (valleys and peaks) and from lobule
borders. Profiles sit at the 40 positions of the SPT model's pattern scan, in the forms of VARIANTS. The group mean is
the mean of the slides' means; SE is between slides, one slide per animal; SD is between lobules within slides,
pooled. Slides open through their sidecars in --slides, chosen by the sidecar's stain. Writes one .npz per slide,
three tables and the run record.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize

from slideviz.analysis import dab, tissue, zonation
from slideviz.analysis.tiling import pick_level_um
from slideviz.analysis.tissue import mask_from_level, pick_level, working_mask
from slideviz.data.provenance import code_version
from slideviz.data.registration import read_sidecar
from slideviz.io.reader import open_slide

LOW_PCT, HIGH_PCT = 5, 99  # percentiles of the slide's profile pixels for the per-slide forms
EDGE_BIN_UM = 50.0  # bins of distance to the section outline for the edge curve
EDGE_MAX_UM = 1500.0  # edge curve range; tissue farther in is the interior
RECORD = "zonation_profiles.json"

VARIANTS = {  # name: (position, values, meaning)
    "local": ("points", "local", "DAB between its 500 um window's low (0) and high (1) level, uncut"),
    "slide": ("points", "raw", f"(DAB - p{LOW_PCT}) / (p{HIGH_PCT} - p{LOW_PCT}) over the slide's profile pixels"),
    "unsubtracted": ("points", "raw", f"DAB / p{HIGH_PCT} over the slide's profile pixels; zero is no stain"),
    "raw_od": ("points", "raw", "DAB optical density"),
    "local_border": ("border", "local", "local, position from lobule borders and portal lumina"),
}
ZONATION_SETTINGS = ("LANDSCAPE_UM", "BLOCK_TISSUE_MIN", "SIGMA_UM", "PEAK_GAP_UM", "EDGE_UM", "LOBULE_BIN_MIN_UM2",
                     "VESSEL_REACH_UM", "VESSEL_MAX_UM2", "WALL_UM")
DAB_SETTINGS = ("WORK_UM", "GLASS_OD", "SMOOTH_UM", "WINDOW_UM", "STEP_UM", "CELL_UM", "LOW_Q", "HIGH_Q",
                "WINDOW_TISSUE_MIN")
TISSUE_SETTINGS = ("LUMEN_OPEN_UM", "LUMEN_MIN_UM2", "RIM_UM")


def scaled(variant: str, x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Raw or local values in a variant's units, from the slide's low and high percentiles."""
    if variant == "slide":
        return (x - low) / (high - low)
    if variant == "unsubtracted":
        return x / high
    return x


def measure(job: tuple) -> dict:
    """One slide's sums per lobule and position, its edge curve and counts; writes <stem>_zonation.npz."""
    stem, slides, out = job
    start = time.time()
    info, levels = open_slide(slides / read_sidecar(slides, stem)["file"])
    shapes = [level.shape for level in levels]
    index = pick_level_um(shapes, info.pixel_size_um, dab.WORK_UM)
    um = info.pixel_size_um * shapes[0][1] / shapes[index][1]
    coarse = np.asarray(levels[pick_level(levels)])
    outline = mask_from_level(coarse)
    od = dab.stain_od(np.asarray(levels[index]), dab.glass_colour(coarse, outline))
    cells = working_mask(outline, dab.glass_like(od), um)
    inside = resize(outline, cells.shape, order=0).astype(bool)
    stain = dab.smooth(od[..., dab.DAB], cells, um)
    del od
    local = zonation.local_share(stain, cells, dab.local_levels(stain, cells, um))
    land = zonation.landscape(stain, cells, inside, um)
    por = zonation.portality(land, cells, inside)
    low, high = (float(v) for v in np.percentile(stain[por.keep], [LOW_PCT, HIGH_PCT]))

    arrays = {}
    for pname, p in (("points", por.points), ("border", por.border)):
        for vname, values in (("raw", stain), ("local", local)):
            prof = zonation.profile(p, values, por.lobule, por.keep)
            arrays[f"{pname}_{vname}_sums"], arrays[f"{pname}_{vname}_counts"] = prof.sums, prof.counts

    # stain against distance to the section outline, padded so a section running off the image keeps its edge
    edge = distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1][cells] * um
    bins = (edge // EDGE_BIN_UM).astype(np.int64)
    del edge
    n = round(EDGE_MAX_UM / EDGE_BIN_UM)
    near = bins < n
    counts = np.bincount(bins[near], minlength=n)
    arrays["edge_counts"] = counts
    for name, values in (("raw", stain[cells]), ("local", local[cells])):
        arrays[f"edge_{name}"] = np.where(counts > 0, np.bincount(bins[near], values[near], n) / np.maximum(counts, 1),
                                          np.nan)
        arrays[f"interior_{name}"] = np.array(values[~near].mean() if (~near).any() else np.nan)

    centre = land.block / 2
    arrays["peaks_um"] = (land.peaks * land.block + centre) * um
    arrays["valleys_um"] = (land.valleys * land.block + centre) * um
    row = {"slide": stem, "um_per_px": round(um, 4), "landscape_um_per_px": round(land.um_per_px, 3),
           "peaks": len(land.peaks), "valleys": len(land.valleys), "lobules": int(por.lobule.max()),
           "tissue_mm2": round(float(cells.sum()) * (um / 1000) ** 2, 3),
           "profile_mm2": round(float(por.keep.sum()) * (um / 1000) ** 2, 3),
           "lobule_min_px": zonation.lobule_min_count(um),
           "mean_p_points": round(float(por.points[por.keep].mean()), 4),
           "mean_p_border": round(float(por.border[por.keep].mean()), 4),
           "low_high": [low, high], "seconds": round(time.time() - start, 1)}
    np.savez_compressed(out / f"{stem}_zonation.npz", **arrays, meta=json.dumps(row))
    print(f"{stem}: {row['peaks']} peaks, {row['valleys']} valleys, {row['lobules']} lobules, "
          f"{row['profile_mm2']} of {row['tissue_mm2']} mm2 profiled, {row['seconds']} s", flush=True)
    return {"row": row, "profiles": {k: v for k, v in arrays.items() if k.endswith(("_sums", "_counts"))},
            "edge": {k: arrays[k] for k in ("edge_counts", "edge_raw", "edge_local")}}


def variant_stats(variant: str, results: list[dict]) -> tuple[list[dict], dict]:
    """Per-slide rows and the group's mean, SE between slides and pooled SD between lobules for one variant."""
    position, values, _ = VARIANTS[variant]
    rows, means, shares = [], [], []
    squares = np.zeros(len(zonation.POSITIONS))
    freedom = np.zeros(len(zonation.POSITIONS))
    lobules = np.zeros(len(zonation.POSITIONS), np.int64)
    for result in results:
        row, stored = result["row"], result["profiles"]
        prof = zonation.Profile(stored[f"{position}_{values}_sums"], stored[f"{position}_{values}_counts"])
        low, high = row["low_high"]
        mean = scaled(variant, prof.mean(), low, high)
        each = scaled(variant, prof.lobule_means(row["lobule_min_px"]), low, high)
        finite = np.isfinite(each)
        n = finite.sum(axis=0)
        centred = np.where(finite, each - np.where(finite, each, 0).sum(axis=0) / np.maximum(n, 1), 0)
        ss = (centred**2).sum(axis=0)
        with np.errstate(all="ignore"):
            sd = np.where(n >= 2, np.sqrt(ss / (n - 1)), np.nan)
        squares += ss
        freedom += np.maximum(n - 1, 0)
        lobules += n
        means.append(mean)
        shares.append(prof.share())
        rows.extend({"variant": variant, "slide": row["slide"], "position": pos, "mean": m, "sd_lobules": s,
                     "n_lobules": int(k), "tissue_share": t}
                    for pos, m, s, k, t in zip(zonation.POSITIONS, mean, sd, n, prof.share()))
    means = np.array(means)
    with np.errstate(all="ignore"):
        group = {"mean": means.mean(axis=0),
                 "sd_lobules": np.where(freedom > 0, np.sqrt(squares / freedom), np.nan),
                 "se_slides": means.std(axis=0, ddof=1) / np.sqrt(len(means)) if len(means) > 1
                 else np.full(len(zonation.POSITIONS), np.nan),
                 "n_lobules": lobules, "tissue_share": np.mean(shares, axis=0)}
    return rows, group


def write_table(path: Path, rows: list[dict]) -> None:
    """Rows as a CSV, floats to 6 significant digits, NaN empty."""
    def cell(v: object) -> object:
        """One value as written."""
        if isinstance(v, float | np.floating):
            return "" if not np.isfinite(v) else f"{v:.6g}"
        return v
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: cell(v) for k, v in row.items()} for row in rows)


def chosen_slides(slides: Path, stain: str, blocks: list[str] | None) -> list[str]:
    """Stems of the slides in `slides` whose sidecar has this stain and whose stem contains one of the blocks."""
    stems, animals = [], {}
    for path in sorted(slides.glob("*.json")):
        stem = path.name.removesuffix(".json").removesuffix(".ome")
        sidecar = json.loads(path.read_text())
        if sidecar.get("stain") != stain or (blocks and not any(block in stem for block in blocks)):
            continue
        animal = tuple(sidecar.get(k) for k in ("species", "substance", "dose_mg_per_kg", "animal_id"))
        if animal in animals:
            raise SystemExit(f"{stem} and {animals[animal]} are one animal; SE between slides needs one slide each")
        animals[animal] = stem
        stems.append(stem)
    return stems


def main() -> None:
    """Measure the chosen slides and write the per-slide files, the tables and the run record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="slides with their sidecars")
    parser.add_argument("--stain", required=True, help="the sidecars' stain, e.g. cyp2e1")
    parser.add_argument("--blocks", nargs="*", help="only stems containing one of these")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")
    stems = chosen_slides(args.slides, args.stain, args.blocks)
    if not stems:
        raise SystemExit(f"no {args.stain} slides in {args.slides} matching {args.blocks}")
    args.out.mkdir(parents=True, exist_ok=True)

    with Pool(max(1, min(args.workers, len(stems)))) as pool:
        results = pool.map(measure, [(stem, args.slides, args.out) for stem in stems])

    slide_rows, group_rows = [], []
    for variant in VARIANTS:
        rows, group = variant_stats(variant, results)
        slide_rows.extend(rows)
        group_rows.extend({"variant": variant, "position": pos, "n_slides": len(results),
                           **{k: v[i] for k, v in group.items()}} for i, pos in enumerate(zonation.POSITIONS))
    write_table(args.out / "zonation_slides.csv", slide_rows)
    write_table(args.out / "zonation_group.csv", group_rows)
    write_table(args.out / "zonation_edge.csv", [
        {"slide": r["row"]["slide"], "distance_um": i * EDGE_BIN_UM, "tissue_px": int(r["edge"]["edge_counts"][i]),
         "raw_od": float(r["edge"]["edge_raw"][i]), "local": float(r["edge"]["edge_local"][i])}
        for r in results for i in range(len(r["edge"]["edge_counts"]))])

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(Path(__file__)),
        "inputs": {"slides": str(args.slides), "stain": args.stain, "blocks": args.blocks,
                   "files": {s: read_sidecar(args.slides, s)["file"] for s in stems}},
        "method": ("DAB optical density (Ruifrok and Johnston 2001) smoothed over cells on the working grid; "
                   "position p = dp / (dp + dc) from slideviz.analysis.zonation, the portality of Albadry et al. "
                   "2024 with valleys of the blurred DAB as portal points and peaks as central points; "
                   "each slide in its own frame, no registration"),
        "positions": "numpy.linspace(0, 1, 40), the SPT model's pattern scan; bins centred on them",
        "variants": {k: v[2] for k, v in VARIANTS.items()},
        "statistics": ("mean: mean of the slides' means, each over its profile pixels; se_slides: SD of the slides' "
                       "means / sqrt(n), one slide per animal; sd_lobules: SD between lobule means within slides, "
                       "pooled over slides; a lobule counts in a bin with at least LOBULE_BIN_MIN_UM2 of tissue"),
        "profile_pixels": ("tissue at least EDGE_UM inside the outline, more than WALL_UM from a lumen, "
                           "on landscape pixels that are mostly tissue"),
        "settings": ({name: getattr(zonation, name) for name in ZONATION_SETTINGS}
                     | {name: getattr(dab, name) for name in DAB_SETTINGS}
                     | {name: getattr(tissue, name) for name in TISSUE_SETTINGS}
                     | {"LOW_PCT": LOW_PCT, "HIGH_PCT": HIGH_PCT, "EDGE_BIN_UM": EDGE_BIN_UM,
                        "EDGE_MAX_UM": EDGE_MAX_UM}),
        "files": {"<slide>_zonation.npz": ("<position>_<values>_sums and _counts per lobule (row 0 outside every "
                                           "lobule) and bin, for points or border and raw or local; edge_counts, "
                                           "edge_raw, edge_local per EDGE_BIN_UM from the outline, interior_raw "
                                           "and interior_local beyond EDGE_MAX_UM; peaks_um and valleys_um, rows "
                                           "and columns from the image's top left; meta, the slide row"),
                  "zonation_slides.csv": "per slide, variant and position: mean, sd_lobules, n_lobules, tissue_share",
                  "zonation_group.csv": "per variant and position: mean, sd_lobules, se_slides, n_lobules, "
                                        "tissue_share",
                  "zonation_edge.csv": "per slide and distance bin: tissue pixels, mean raw DAB and local share"},
        "slides": [r["row"] for r in results],
    }
    (args.out / RECORD).write_text(json.dumps(record, indent=1) + "\n")
    print(f"=== {len(results)} slides; tables and {RECORD} in {args.out} ===")


if __name__ == "__main__":
    main()
