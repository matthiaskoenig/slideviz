"""Polygons to triangle meshes for DOLFINx: gmsh with named areas and edges and a size field, per variant and size.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import gmsh
import numpy as np
import shapely
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from polygons import code_version, sha256
from shapely.geometry import shape

AREA_TAGS = {"surviving": 1, "necrotic": 2, "tissue": 3}
EDGE_TAGS = {"vessel_wall": 11, "section_edge": 12, "field_cut": 13, "necrosis_border": 14}
REFINED = ("vessel_wall", "section_edge", "necrosis_border")  # the square's cut sides are artificial
DIST_MIN, DIST_MAX = 2, 10  # small size up to DIST_MIN x small from a refined edge, large size from DIST_MAX x large
SAMPLING = 20  # points per edge piece in gmsh's distance field
EPS_UM = 1e-6


def frame(meta: dict, shape_rc: tuple[int, int]) -> tuple[float, float, float, float]:
    """The square's outer pixel edges (x0, y0, x1, y1) in um."""
    um, (oy, ox) = meta["um_per_px"], meta["origin_um"]
    return ox - um / 2, oy - um / 2, ox + (shape_rc[1] - 0.5) * um, oy + (shape_rc[0] - 0.5) * um


def on_side(a: tuple, b: tuple, box: tuple) -> bool:
    """An edge piece lying on one of the square's sides."""
    x0, y0, x1, y1 = box
    return any(abs(a[i] - v) < EPS_UM and abs(b[i] - v) < EPS_UM for i, v in ((0, x0), (0, x1), (1, y0), (1, y1)))


def lumen_beside(a: tuple, b: tuple, lumen: np.ndarray, meta: dict) -> bool:
    """The lumen mask one pixel to either side of an edge piece's midpoint."""
    um, (oy, ox) = meta["um_per_px"], meta["origin_um"]
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = np.hypot(dx, dy) or 1.0
    for side in (1, -1):
        r = round((my + side * dx / length * um - oy) / um)
        c = round((mx - side * dy / length * um - ox) / um)
        if 0 <= r < lumen.shape[0] and 0 <= c < lumen.shape[1] and lumen[r, c]:
            return True
    return False


def geometry(items: list[tuple[str, object]]) -> tuple[dict, dict, set, dict]:
    """gmsh points, edge pieces and surfaces, each corner and edge made once; surfaces by name, edges, hole edges."""
    g = gmsh.model.geo
    points, lines, hole_lines, surfaces = {}, {}, set(), {}

    def point(xy: tuple) -> int:
        """The gmsh point at xy, made on first use."""
        if xy not in points:
            points[xy] = g.addPoint(xy[0], xy[1], 0)
        return points[xy]

    def loop(coords, hole: bool) -> int:
        """A curve loop over a ring, its edge pieces shared with every area that touches them."""
        ids = [point(tuple(c)) for c in coords[:-1]]
        signed = []
        for a, b in zip(ids, ids[1:] + ids[:1], strict=True):
            key = (min(a, b), max(a, b))
            if key not in lines:
                lines[key] = g.addLine(*key)
            if hole:
                hole_lines.add(lines[key])
            signed.append(lines[key] if a == key[0] else -lines[key])
        return g.addCurveLoop(signed)

    for name, polygon in items:
        for part in shapely.get_parts(polygon):
            loops = [loop(part.exterior.coords, False)] + [loop(h.coords, True) for h in part.interiors]
            surfaces.setdefault(name, []).append(g.addPlaneSurface(loops))
    g.synchronize()
    return surfaces, lines, hole_lines, {tag: xy for xy, tag in points.items()}


def edge_groups(lines: dict, hole_lines: set, xy: dict, box: tuple, lumen: np.ndarray, meta: dict) -> dict:
    """Edge pieces by name: between two areas, on the square's sides, around a vessel, or the section's edge."""
    groups = {name: [] for name in EDGE_TAGS}
    for (a, b), tag in lines.items():
        up, _ = gmsh.model.getAdjacencies(1, tag)
        if len(up) == 2:
            groups["necrosis_border"].append(tag)
        elif on_side(xy[a], xy[b], box):
            groups["field_cut"].append(tag)
        elif tag in hole_lines or lumen_beside(xy[a], xy[b], lumen, meta):
            groups["vessel_wall"].append(tag)
        else:
            groups["section_edge"].append(tag)
    return groups


def size_field(groups: dict, small: float, large: float) -> None:
    """Triangles of `small` um at the refined edges, growing linearly to `large` um further in."""
    field = gmsh.model.mesh.field
    distance = field.add("Distance")
    field.setNumbers(distance, "CurvesList", [t for name in REFINED for t in groups[name]])
    field.setNumber(distance, "Sampling", SAMPLING)
    threshold = field.add("Threshold")
    field.setNumber(threshold, "InField", distance)
    for key, value in (("SizeMin", small), ("SizeMax", large), ("DistMin", DIST_MIN * small),
                       ("DistMax", DIST_MAX * large)):
        field.setNumber(threshold, key, value)
    field.setAsBackgroundMesh(threshold)
    for option in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(option, 0)


def view_arrays(surfaces: dict, groups: dict) -> dict:
    """The mesh as plain arrays: corner points in um, triangles with their area tag, named edge segments."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(node_tags)}
    triangles, area_tag, segments, edge_tag = [], [], [], []
    for name, tags in surfaces.items():
        for surface in tags:
            _, _, nodes = gmsh.model.mesh.getElements(2, surface)
            tri = np.array([index[int(n)] for n in nodes[0]]).reshape(-1, 3)
            triangles.append(tri)
            area_tag.append(np.full(len(tri), AREA_TAGS[name]))
    for name, tags in groups.items():
        for curve in tags:
            _, _, nodes = gmsh.model.mesh.getElements(1, curve)
            seg = np.array([index[int(n)] for n in nodes[0]]).reshape(-1, 2)
            segments.append(seg)
            edge_tag.append(np.full(len(seg), EDGE_TAGS[name]))
    return {"points_um": coords.reshape(-1, 3)[:, :2], "triangles": np.concatenate(triangles),
            "triangle_area_tag": np.concatenate(area_tag), "edges": np.concatenate(segments),
            "edge_tag": np.concatenate(edge_tag)}


def tag_counts(values: np.ndarray) -> dict[str, int]:
    """How many cells or facets carry each tag."""
    return {str(int(v)): int((values == v).sum()) for v in np.unique(values)}


def read_back(path: Path) -> dict:
    """Cells and tag counts of a written XDMF mesh, read as the solver reads it."""
    with XDMFFile(MPI.COMM_WORLD, path, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        mesh.topology.create_connectivity(1, 2)
        areas, edges = xdmf.read_meshtags(mesh, name="areas"), xdmf.read_meshtags(mesh, name="edges")
    return {"cells": int(mesh.topology.index_map(2).size_local),
            "areas": tag_counts(areas.values), "edges": tag_counts(edges.values)}


def build(variant: str, polygons: list, lumen: np.ndarray, meta: dict, small: float, large: float,
          base: Path) -> dict:
    """One mesh: geometry, names, size field, triangles, DOLFINx import, written files; its checks."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(base.name)
    if variant == "follow":
        items = polygons
    else:
        items = [("tissue", p) for p in shapely.get_parts(shapely.union_all([p for _, p in polygons]))]
    surfaces, lines, hole_lines, xy = geometry(items)
    groups = edge_groups(lines, hole_lines, xy, frame(meta, lumen.shape), lumen, meta)
    for name, tags in surfaces.items():
        gmsh.model.addPhysicalGroup(2, tags, AREA_TAGS[name], name)
    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(1, tags, EDGE_TAGS[name], name)
    size_field(groups, small, large)
    start = time.time()
    gmsh.model.mesh.generate(2)
    seconds = time.time() - start

    _, elements, _ = gmsh.model.mesh.getElements(2)
    quality = np.asarray(gmsh.model.mesh.getElementQualities(np.concatenate(elements), "minSICN"))
    arrays = view_arrays(surfaces, groups)
    p, t = arrays["points_um"], arrays["triangles"]
    u, v = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]
    tri_area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    gmsh.write(str(base.with_suffix(".msh")))

    data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    mesh = data.mesh
    mesh.topology.create_connectivity(1, 2)
    data.cell_tags.name, data.facet_tags.name = "areas", "edges"
    with XDMFFile(MPI.COMM_WORLD, base.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(data.cell_tags, mesh.geometry)
        xdmf.write_meshtags(data.facet_tags, mesh.geometry)
    np.savez_compressed(base.parent / f"{base.name}_view.npz", **arrays)
    written = {"cells": int(mesh.topology.index_map(2).size_local),
               "areas": tag_counts(data.cell_tags.values), "edges": tag_counts(data.facet_tags.values)}

    return {
        "triangles": len(quality),
        "nodes": len(p),
        "seconds": round(seconds, 2),
        "quality_min": round(float(quality.min()), 3),
        "quality_median": round(float(np.median(quality)), 3),
        "quality_below_0.3_pct": round(100 * float((quality < 0.3).mean()), 3),
        "edge_pieces": {name: len(tags) for name, tags in groups.items()},
        "area_um2": {name: round(float(tri_area[arrays["triangle_area_tag"] == AREA_TAGS[name]].sum()), 1)
                     for name in surfaces},
        "dolfinx_cells": int(mesh.topology.index_map(2).size_local),
        "dolfinx_cell_tags": sorted(int(v) for v in set(data.cell_tags.values)),
        "dolfinx_facet_tags": sorted(int(v) for v in set(data.facet_tags.values)),
        "dolfinx_names": sorted(data.physical_groups),
        "xdmf_read_back_matches": read_back(base.with_suffix(".xdmf")) == written,
    }


def main() -> None:
    """Write one mesh per variant and size, each with its record, and print their checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--polygons", type=Path, required=True, help="a polygons.py GeoJSON")
    parser.add_argument("--variant", nargs="+", choices=("follow", "plain"), required=True)
    parser.add_argument("--size-um", nargs="+", required=True, metavar="SMALL:LARGE",
                        help="triangle size at the refined edges and far inside, in um")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")

    collection = json.loads(args.polygons.read_text())
    meta = collection["meta"]
    with np.load(meta["inputs"]["file"]) as stored:
        lumen = stored["lumen"]
    polygons = [(f["properties"]["class"], shape(f["geometry"])) for f in collection["features"]
                if f["properties"]["class"] != "domain"]
    polygon_area = {name: round(sum(p.area for k, p in polygons if k == name), 1) for name in ("surviving", "necrotic")}

    args.out.mkdir(parents=True, exist_ok=True)
    for variant in args.variant:
        for size in args.size_um:
            small, large = (float(v) for v in size.split(":"))
            base = args.out / f"{meta['animal']}_{variant}_h{small:g}-{large:g}um"
            checks = build(variant, polygons, lumen, meta, small, large, base)
            record = {
                "slide": meta["slide"],
                "animal": meta["animal"],
                "polygons": {"file": str(args.polygons), "sha256": sha256(args.polygons), "code": meta["code"],
                             "tolerance_um": meta["tolerance_um"], "area_um2": polygon_area},
                "variant": variant,
                "size_um": {"small": small, "large": large, "dist_min_um": DIST_MIN * small,
                            "dist_max_um": DIST_MAX * large, "refined": list(REFINED)},
                "tags": {"areas": AREA_TAGS, "edges": EDGE_TAGS},
                "coordinates": meta["coordinates"],
                "files": [f"{base.name}{s}" for s in (".xdmf", ".h5", ".msh", "_view.npz")],
                "checks": checks,
                "code": code_version(),
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            base.with_suffix(".json").write_text(json.dumps(record, indent=1))
            print(f"{base.name}: {json.dumps(checks)}", flush=True)


if __name__ == "__main__":
    main()
