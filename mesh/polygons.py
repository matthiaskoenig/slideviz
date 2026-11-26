"""Masks to polygons for gmsh: a mesh_inputs.npz field or section as surviving and necrotic areas, outlines in um.

Example, on gpu1:
    cd ~/Projects/slideviz/mesh && ~/.pixi/bin/pixi run --locked python polygons.py \
        --inputs /data/michelle/mesh_281mg_m1_field_2026-10-25/inputs/281mg_m1_mesh_inputs.npz \
        --tolerance-um 1 2 4 8 --out /data/michelle/mesh_281mg_m1_field_2026-10-25/polygons

Writes one GeoJSON per tolerance: the domain (all tissue to mesh, lumina and the outside as holes) and the
surviving and necrotic areas, simplified together so their shared borders stay shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import shapely
from scipy.ndimage import binary_dilation
from shapely import box, coverage_simplify, orient_polygons, segmentize, union_all
from skimage.measure import label, regionprops

NONE, SURVIVING, NECROTIC = 0, 1, 2
NAMES = {NONE: "none", SURVIVING: "surviving", NECROTIC: "necrotic"}
OUT_NAME = "{animal}_polygons_{tolerance:g}um.geojson"
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)  # pixels touching by an edge, as the polygons join them


def code_version() -> str:
    """The commit this script ran from, marked dirty for a changed tree."""
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                           check=False, cwd=Path(__file__).resolve().parent)
    return found.stdout.strip() or "unknown"


def sha256(path: Path) -> str:
    """Hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classes(layers: dict) -> np.ndarray:
    """Per pixel: NECROTIC or SURVIVING inside the tissue without lumina, NONE elsewhere."""
    domain = layers["tissue"] & ~layers["lumen"]
    return np.where(domain, np.where(layers["necrotic"], NECROTIC, SURVIVING), NONE).astype(np.uint8)


def to_um(row: float, col: float, origin_um: list[float], um: float) -> list[float]:
    """A pixel position as (x, y) in um on the H&E, x along the columns."""
    return [round(origin_um[1] + col * um, 3), round(origin_um[0] + row * um, 3)]


def window(region, shape: tuple[int, int]) -> tuple[slice, slice, np.ndarray]:
    """A region's bounding box grown by one pixel within the image, and the region as a mask in it."""
    r0, c0, r1, c1 = region.bbox
    r0, c0, r1, c1 = max(r0 - 1, 0), max(c0 - 1, 0), min(r1 + 1, shape[0]), min(c1 + 1, shape[1])
    piece = np.zeros((r1 - r0, c1 - c0), bool)
    piece[region.coords[:, 0] - r0, region.coords[:, 1] - c0] = True
    return slice(r0, r1), slice(c0, c1), piece


def tidy(cls: np.ndarray, min_px: int, origin_um: list[float], um: float) -> tuple[np.ndarray, list[dict]]:
    """Tissue pieces under min_px take the class of the large tissue around them, decided on the input; the changes."""
    small = np.zeros(cls.shape, bool)
    for value in (SURVIVING, NECROTIC):
        labels = label(cls == value, connectivity=1)
        sizes = np.bincount(labels.ravel())
        sizes[0] = min_px
        small |= sizes[labels] < min_px
    out = cls.copy()
    for region in regionprops(label(small, connectivity=1)):
        rows, cols, piece = window(region, cls.shape)
        ring = binary_dilation(piece, FOUR) & ~piece & ~small[rows, cols]
        around = cls[rows, cols][ring]
        counts = {k: int((around == k).sum()) for k in (SURVIVING, NECROTIC)}
        if any(counts.values()):
            target = max(counts, key=counts.get)
        elif region.area >= min_px:  # small pieces that together make a fragment of their own
            inside = cls[rows, cols][piece]
            target = SURVIVING if (inside == SURVIVING).sum() >= (inside == NECROTIC).sum() else NECROTIC
        else:
            target = NONE
        out[rows, cols][piece] = target
    changed = []
    for region in regionprops(label(out != cls, connectivity=1)):
        before = cls[region.coords[:, 0], region.coords[:, 1]]
        changed.append({"from": {NAMES[k]: int((before == k).sum()) for k in (SURVIVING, NECROTIC) if (before == k).any()},
                        "to": NAMES[int(out[tuple(region.coords[0])])], "area_um2": round(region.area * um**2, 1),
                        "centre_um": to_um(*region.centroid, origin_um, um)})
    return out, changed


def diagonals(cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2 x 2 windows whose main or anti diagonal is one class with both other pixels of another class."""
    a, b, c, d = cls[:-1, :-1], cls[:-1, 1:], cls[1:, :-1], cls[1:, 1:]
    return (a == d) & (b != a) & (c != a), (b == c) & (a != b) & (d != b)


def join_diagonals(cls: np.ndarray, passes: int = 100) -> tuple[np.ndarray, int]:
    """Pixels of one class touching only at a corner joined by an edge, so no two rings share a single point."""
    cls = cls.copy()
    joined = 0
    for _ in range(passes):
        main, anti = diagonals(cls)
        if not main.any() and not anti.any():
            return cls, joined
        for r, c in np.argwhere(main):
            if cls[r, c] == cls[r + 1, c + 1] and cls[r, c + 1] != cls[r, c] and cls[r + 1, c] != cls[r, c]:
                cls[r, c + 1] = cls[r, c]
                joined += 1
        for r, c in np.argwhere(anti):
            if cls[r, c + 1] == cls[r + 1, c] and cls[r, c] != cls[r, c + 1] and cls[r + 1, c + 1] != cls[r, c + 1]:
                cls[r, c] = cls[r, c + 1]
                joined += 1
    raise RuntimeError(f"corner contacts left after {passes} passes")


def pixel_polygons(mask: np.ndarray, origin_um: list[float], um: float) -> list:
    """A mask's areas along its pixels' edges in um, a point at every pixel corner, so neighbours share their points."""
    boxes = []
    for r in range(mask.shape[0]):
        steps = np.diff(np.concatenate([[0], mask[r].astype(np.int8), [0]]))
        boxes += [box(s - 0.5, r - 0.5, e - 0.5, r + 0.5)
                  for s, e in zip(np.flatnonzero(steps == 1), np.flatnonzero(steps == -1), strict=True)]
    if not boxes:
        return []
    # in pixel units every corner is a half-integer, then one transform for both classes keeps them bit-identical
    parts = segmentize(shapely.get_parts(union_all(boxes)), 1.0)
    return list(shapely.transform(parts, lambda xy: xy * um + np.array([origin_um[1], origin_um[0]])))


def features(polygons: list, kinds: list[int]) -> list[dict]:
    """GeoJSON features: the domain as one area, then each surviving and necrotic polygon, outer rings anticlockwise."""
    domain = orient_polygons(union_all(polygons))
    out = [{"type": "Feature", "properties": {"class": "domain"}, "geometry": shapely.geometry.mapping(domain)}]
    out += [{"type": "Feature", "properties": {"class": NAMES[k]}, "geometry": shapely.geometry.mapping(orient_polygons(p))}
            for p, k in zip(polygons, kinds, strict=True)]
    return out


def main() -> None:
    """Write the polygons at each tolerance and print their checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True, help="a mesh_inputs.py file")
    parser.add_argument("--tolerance-um", type=float, nargs="+", required=True,
                        help="simplification tolerances; 0 keeps the pixel edges")
    parser.add_argument("--min-piece-um2", type=float, default=100.0,
                        help="tissue pieces under this take the class of the large tissue around them")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")

    with np.load(args.inputs) as stored:
        layers = {name: stored[name] for name in stored.files if name != "meta"}
        source = json.loads(str(stored["meta"]))
    um, origin = source["um_per_px"], source["origin_um"]
    cls, changed = tidy(classes(layers), int(np.ceil(args.min_piece_um2 / um**2)), origin, um)
    cls, joined = join_diagonals(cls)

    polygons, kinds = [], []
    for value in (SURVIVING, NECROTIC):
        parts = pixel_polygons(cls == value, origin, um)
        polygons += parts
        kinds += [value] * len(parts)
    pixel_um2 = float((cls != NONE).sum() * um**2)

    args.out.mkdir(parents=True, exist_ok=True)
    for tolerance in args.tolerance_um:
        shapes = polygons if tolerance == 0 else list(coverage_simplify(polygons, tolerance))
        area = sum(p.area for p in shapes)
        necrotic = sum(p.area for p, k in zip(shapes, kinds, strict=True) if k == NECROTIC)
        checks = {
            "polygons": {NAMES[k]: kinds.count(k) for k in (SURVIVING, NECROTIC)},
            "points": int(sum(shapely.get_num_coordinates(p) for p in shapes)),
            "points_pixel_edges": int(sum(shapely.get_num_coordinates(p) for p in polygons)),
            "area_um2": round(area, 1),
            "area_pixels_um2": round(pixel_um2, 1),
            "necrotic_pct": round(100 * necrotic / area, 2),
            "necrotic_pixels_pct": round(100 * float((cls == NECROTIC).sum()) / max(int((cls != NONE).sum()), 1), 2),
            "all_valid": bool(all(p.is_valid for p in shapes)),
            "coverage_valid": bool(shapely.coverage_is_valid(shapes)),
            "corner_contacts_left": int(sum(w.sum() for w in diagonals(cls))),
            "smallest_um2": round(min(p.area for p in shapes), 1),
        }
        meta = {
            "slide": source["slide"],
            "animal": source["animal"],
            "inputs": {"file": str(args.inputs), "sha256": sha256(args.inputs), "code": source["code"]},
            "coordinates": "x along the columns, y along the rows, um on the H&E level-0 frame",
            "um_per_px": um,
            "origin_um": origin,
            "tolerance_um": tolerance,
            "min_piece_um2": args.min_piece_um2,
            "changed_pieces": changed,
            "corner_pixels_joined": joined,
            "checks": checks,
            "code": code_version(),
            "written": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = args.out / OUT_NAME.format(animal=source["animal"], tolerance=tolerance)
        path.write_text(json.dumps({"type": "FeatureCollection", "meta": meta, "features": features(shapes, kinds)}))
        print(f"{path.name}: {json.dumps(checks)}", flush=True)
    moved = {name: sum(c["from"].get(name, 0) for c in changed) for name in ("surviving", "necrotic")}
    print(f"tidy under {args.min_piece_um2:g} um2: {len(changed)} pieces, {sum(c['area_um2'] for c in changed):.0f} um2 "
          f"({moved['necrotic']} necrotic and {moved['surviving']} surviving px changed); "
          f"corner contacts joined: {joined} px", flush=True)


if __name__ == "__main__":
    main()
