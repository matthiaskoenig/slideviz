"""Necrosis labels per encoder patch, 28 x 28 per training tile, from the polygons behind the tile labels.

    uv run python scripts/patch_labels.py --tiles /data/michelle/mouse/tiles --slides /data/michelle/slides \
        --annotations /data/michelle/slides/annotations/cvat_polygons.json \
        --overviews /data/michelle/slides/annotation_overviews \
        --lumen /data/michelle/necrosis_lumen_2026-10-11 --out <new dir> --workers 10

The encoder sees a whole tile resized to 224 px and describes it in 28 x 28 patches, so patch (r, c)
covers the tile's rows r * size / 28 to (r + 1) * size / 28, edges that fall between pixels (6.07 px at
170 px). Tissue is the tissue outline without lumina (the 11 Oct lumen rule). A patch is necrotic when
at least half of it is tissue inside a polygon, not necrotic when it is at least half tissue and its
centre lies more than --band-um outside every polygon, and left out of training otherwise: the band
between, since the polygons stop short of the necrosis edge, and patches mostly glass or lumen.
The fractions behind the labels are written beside them, and the labels are set from those stored
values, so a different rule needs no rerun. Every tile's patches must average back to its label in
the manifest, or nothing is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from slideviz.analysis.annotation import (
    cvat_polygons,
    rasterise,
    read_overview,
    to_level,
)
from slideviz.analysis.tiling import resampled
from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.data.provenance import code_version
from slideviz.io.reader import open_slide

GRID = 28  # the encoder's patch grid, as in embed_tiles.py's patches.npy
BAND_UM = 30.0  # a guess at how far the polygons stop short of the necrosis edge
HALF = 0.5  # the share of a patch that decides it
LEFT_OUT = -1

# the manifest rounds each tile's fractions to 4 decimals
TOLERANCE = 5e-5 + 1e-6

# the stored fractions and distances, which the labels are set from
STORED = np.float16


def patch_weights(size_px: int, grid: int = GRID) -> np.ndarray:
    """Share of each tile pixel row in each patch row, so W @ window @ W.T is the mean inside every patch."""
    edges = np.arange(grid + 1) * size_px / grid
    pixels = np.arange(size_px)[None, :]
    overlap = np.minimum(edges[1:, None], pixels + 1) - np.maximum(edges[:-1, None], pixels)
    return np.clip(overlap, 0, None) / (size_px / grid)


def one_slide(job: tuple) -> dict:
    """Patch fractions and centre distances for every manifest tile of one slide."""
    slide, rows, polygons, overview, slides, lumen_dir = job
    info, levels = open_slide(slides / f"{slide}.ome.tiff")
    (level,) = {r["level"] for r in rows}
    (size,) = {r["size_px"] for r in rows}
    shape = levels[level].shape[:2]
    um_per_px = info.pixel_size_um * (levels[0].shape[1] / shape[1])

    # the tile labels' polygon mask, built as annotation.label_grid builds it
    on_level = [to_level(cvat_polygons(p), overview["um_per_px"], um_per_px) for p in polygons]
    mask = rasterise(on_level, (round(overview["height"] * overview["um_per_px"] / um_per_px),
                                round(overview["width"] * overview["um_per_px"] / um_per_px)))
    drawn_any = bool(mask.any())
    distance = distance_transform_edt(~mask).astype(np.float32) * um_per_px if drawn_any else None

    # the tissue outline as tiling.build_grid reads it, and the lumina the necrotic areas leave out
    outline = mask_from_level(np.asarray(levels[pick_level(levels)]))
    stored = np.load(lumen_dir / f"{rows[0]['animal']}_lumen.npz")
    lumen_meta = json.loads(str(stored["meta"]))
    lumen = stored["lumen"]
    if lumen_meta["slide"] != slide or lumen.shape not in {lv.shape[:2] for lv in levels}:
        raise ValueError(f"{lumen_dir}: the lumen file for {rows[0]['animal']} matches no level of {slide}")

    weights = patch_weights(size)
    centres = ((np.arange(GRID) + 0.5) * size / GRID).astype(int)
    inside = np.zeros((len(rows), GRID, GRID), np.float32)
    inside_tissue, tissue = np.zeros_like(inside), np.zeros_like(inside)
    near = np.full_like(inside, np.inf)
    label_error, tissue_error, edge = 0.0, 0.0, 0
    for i, row in enumerate(rows):
        y, x = row["y"], row["x"]
        footprint = np.zeros((size, size), np.float32)
        part = mask[y:y + size, x:x + size]
        footprint[:part.shape[0], :part.shape[1]] = part  # past the mask's edge holds no polygon
        inside[i] = weights @ footprint @ weights.T

        drawn = resampled(outline, shape, y, x, size)
        tissue_px = (drawn & ~resampled(lumen, shape, y, x, size)).astype(np.float32)
        tissue[i] = weights @ tissue_px @ weights.T
        inside_tissue[i] = weights @ (footprint * tissue_px) @ weights.T
        tissue_error = max(tissue_error, abs(float(drawn.mean()) - row["tissue"]))

        if part.shape == (size, size):
            label_error = max(label_error, abs(float(inside[i].mean()) - row["necrosis"]))
        else:
            edge += 1  # label_grid averaged only the part inside the mask
        if drawn_any:
            near[i] = distance[np.ix_((y + centres).clip(0, distance.shape[0] - 1),
                                      (x + centres).clip(0, distance.shape[1] - 1))]

    print(f"{slide}: {len(rows):,} tiles, {len(polygons)} polygons, label check {label_error:.2e}, "
          f"tissue check {tissue_error:.2e}, {edge} tiles at the mask edge", flush=True)
    return {"slide": slide, "inside": inside, "inside_tissue": inside_tissue, "tissue": tissue,
            "distance": near, "label_error": label_error, "tissue_error": tissue_error, "edge_tiles": edge,
            "level": level, "size_px": size, "um_per_px": um_per_px,
            "lumen": {"code": lumen_meta["code"], "written": lumen_meta["written"]}}


def main() -> None:
    """Label every training tile's patches, check them against the tile labels, and write them with the rule."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles", type=Path, required=True, help="export_tiles.py output the embeddings came from")
    parser.add_argument("--slides", type=Path, required=True, help="H&E OME-TIFFs")
    parser.add_argument("--annotations", type=Path, required=True, help="the CVAT polygon export")
    parser.add_argument("--overviews", type=Path, required=True, help="the annotated overviews' sidecars")
    parser.add_argument("--lumen", type=Path, required=True, help="necrosis_lumen.py output, one mask per slide")
    parser.add_argument("--out", type=Path, required=True, help="a new folder; one that holds labels is refused")
    parser.add_argument("--band-um", type=float, default=BAND_UM)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if (args.out / "patch_labels_meta.json").exists():
        raise SystemExit(f"{args.out} already holds patch labels; give a new --out")

    manifest_bytes = (args.tiles / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = manifest["tiles"]
    annotation_bytes = args.annotations.read_bytes()
    export = json.loads(annotation_bytes)
    empty = set(manifest["confirmed_empty"])

    by_slide: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_slide.setdefault(row["slide"], []).append(i)

    jobs = []
    for slide, indices in by_slide.items():
        key = f"{slide}_s0.png"
        if key not in export and key not in empty:
            raise SystemExit(f"{slide} has tiles but neither polygons nor a confirmed-empty entry")
        overview = read_overview(args.overviews / f"{slide}_s0.json")
        if key in export and (overview["width"], overview["height"]) != (export[key]["width"], export[key]["height"]):
            raise SystemExit(f"{slide}: CVAT and the overview sidecar disagree on the overview size")
        polygons = [p["points"] for p in export[key]["polygons"]] if key in export else []
        jobs.append((slide, [rows[i] for i in indices], polygons, overview, args.slides, args.lumen))

    # a worker the system kills stops the run with an error rather than leaving it waiting
    with ProcessPoolExecutor(max(1, min(args.workers, len(jobs)))) as pool:
        results = list(pool.map(one_slide, jobs))

    # nothing is written unless every tile's patches reproduce the labels it was trained on
    for check in ("label_error", "tissue_error"):
        worst = max(results, key=lambda r: r[check])
        if worst[check] > TOLERANCE:
            raise SystemExit(f"patches do not reproduce the manifest's {check.removesuffix('_error')}: "
                             f"{worst[check]:.2e} on {worst['slide']}, nothing written")

    shape = (len(rows), GRID, GRID)
    arrays = {name: np.zeros(shape, STORED) for name in ("inside", "inside_tissue", "tissue", "distance")}
    for result in results:
        for name, array in arrays.items():
            array[by_slide[result["slide"]]] = result[name]

    labels = np.full(shape, LEFT_OUT, np.int8)
    labels[(arrays["tissue"] >= HALF) & (arrays["distance"] > args.band_um)] = 0
    labels[arrays["inside_tissue"] >= HALF] = 1

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "patch_labels.npy", labels)
    for name, array in arrays.items():
        np.save(args.out / f"patch_{name}{'_um' if name == 'distance' else ''}.npy", array)
    (args.out / "patch_labels_tiles.txt").write_text("\n".join(r["tile"] for r in rows) + "\n")

    counts = {animal: Counter() for animal in manifest["animals"]}
    for indices in by_slide.values():
        values, n = np.unique(labels[indices], return_counts=True)
        counts[rows[indices[0]]["animal"]].update(dict(zip(values.tolist(), n.tolist(), strict=True)))
    named = {1: "necrotic", 0: "not_necrotic", LEFT_OUT: "left_out"}
    meta = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(Path(__file__)),
        "rule": (f"necrotic: at least {HALF} of the patch is tissue inside a polygon; not necrotic: at least "
                 f"{HALF} tissue and the patch centre more than {args.band_um} um outside every polygon; left "
                 f"out: everything else (the band between, glass, lumina). Tissue is the outline without "
                 f"lumina. Set from the stored float16 files."),
        "band_um": args.band_um,
        "grid": [GRID, GRID],
        "values": {"1": "necrotic", "0": "not necrotic", str(LEFT_OUT): "left out"},
        "files": {
            "patch_labels.npy": "int8, tiles x row x col, the rule above",
            "patch_inside.npy": "float16, share of the patch inside a polygon (averages to the tile label)",
            "patch_inside_tissue.npy": "float16, share of the patch that is tissue inside a polygon",
            "patch_tissue.npy": "float16, share of the patch inside the tissue outline and outside lumina",
            "patch_distance_um.npy": "float16, patch centre to the nearest polygon pixel in um, 0 inside, "
                                     "inf on a slide without polygons",
            "patch_labels_tiles.txt": "the manifest's tile order, which embeddings.npy and patches.npy share",
        },
        "geometry": {
            "patch_order": "row r from the tile's top, column c from its left, edges at r * size_px / 28",
            "input": "the whole tile resized to the encoder input, no crop, as embed_tiles.py does",
            "level": sorted({r["level"] for r in results}),
            "size_px": sorted({r["size_px"] for r in results}),
            "um_per_px": {r["slide"]: r["um_per_px"] for r in results},
        },
        "n_tiles": len(rows),
        "counts": {named[k]: int((labels == k).sum()) for k in named},
        "counts_per_animal": {a: {named[k]: c.get(k, 0) for k in named} for a, c in counts.items()},
        "check": {
            "label_max_error": max(r["label_error"] for r in results),
            "tissue_max_error": max(r["tissue_error"] for r in results),
            "edge_tiles_not_checked": sum(r["edge_tiles"] for r in results),
            "tolerance": TOLERANCE,
        },
        "inputs": {
            "manifest": str(args.tiles / "manifest.json"),
            "manifest_written": manifest["written"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "annotations": str(args.annotations),
            "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "manifest_annotations": manifest["annotations"],  # where the tile labels were made from
            "confirmed_empty": sorted(empty),
            "overviews": str(args.overviews),
            "lumen": str(args.lumen),
            "lumen_files": {r["slide"]: r["lumen"] for r in results},
            "slides": str(args.slides),
        },
    }
    (args.out / "patch_labels_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(rows):,} tiles: {meta['counts']}, written to {args.out}")


if __name__ == "__main__":
    main()
