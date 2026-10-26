"""Image maps onto build_mesh.py meshes and back onto the pixels: CYP2E1 two ways, necrosis per triangle, the losses."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import ufl
from build_mesh import AREA_TAGS
from dolfinx import fem, geometry
from dolfinx.io import XDMFFile
from dolfinx.mesh import compute_midpoints
from mpi4py import MPI
from polygons import code_version, sha256
from scipy.ndimage import distance_transform_edt, map_coordinates

NO_VALUE, POSITION_SCALE, STEPS = 255, 250, 5  # the CYP2E1 map's codes and gradient steps (slideviz analysis/dab.py)
NECROTIC_SHARE = 0.5  # a triangle is necrotic when at least this share of its pixels is


def read_inputs(path: Path) -> tuple[dict, dict]:
    """The mesh_inputs.py layers and record."""
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files if name != "meta"}, json.loads(str(stored["meta"]))


def filled_cyp2e1(layers: dict) -> tuple[np.ndarray, np.ndarray]:
    """The CYP2E1 position with every pixel taking its nearest measured value, and where it was measured."""
    domain = layers["tissue"] & ~layers["lumen"]
    measured = (layers["cyp2e1"] != NO_VALUE) & domain
    nearest = distance_transform_edt(~measured, return_distances=False, return_indices=True)
    return layers["cyp2e1"].astype(np.float64)[nearest[0], nearest[1]], measured


def read_mesh(path: Path):
    """A build_mesh.py mesh and its area tags, read as the solver reads them."""
    with XDMFFile(MPI.COMM_WORLD, str(path), "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        mesh.topology.create_connectivity(1, 2)
        areas = xdmf.read_meshtags(mesh, name="areas")
    return mesh, areas


def cells_of(mesh, points: np.ndarray) -> np.ndarray:
    """The cell holding each point (x, y, 0) in um, -1 outside the mesh."""
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    found = geometry.compute_colliding_cells(mesh, geometry.compute_collisions_points(tree, points), points)
    counts = np.diff(found.offsets)
    cells = np.full(len(points), -1, dtype=np.int32)
    cells[counts > 0] = found.array[found.offsets[:-1][counts > 0]]
    return cells


def to_pixel(xy: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Positions (x, y) in um as fractional (row, col) on the inputs grid."""
    um, (oy, ox) = meta["um_per_px"], meta["origin_um"]
    return (xy[:, 1] - oy) / um, (xy[:, 0] - ox) / um


def at_corners(mesh, image: np.ndarray, meta: dict) -> fem.Function:
    """An image as one value per corner point, read between the four nearest pixels, linear inside each triangle."""
    space = fem.functionspace(mesh, ("Lagrange", 1))
    field = fem.Function(space)
    rows, cols = to_pixel(space.tabulate_dof_coordinates()[:, :2], meta)
    field.x.array[:] = map_coordinates(image, [rows, cols], order=1, mode="nearest")
    return field


def per_triangle(mesh, cells: np.ndarray, values: np.ndarray, image: np.ndarray, meta: dict) -> fem.Function:
    """An image as one value per triangle: the mean of its pixel centres, the pixel at its middle when it holds none."""
    space = fem.functionspace(mesh, ("DG", 0))
    field = fem.Function(space)
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    inside = cells >= 0
    count = np.bincount(cells[inside], minlength=n)
    mean = np.bincount(cells[inside], weights=values[inside], minlength=n) / np.maximum(count, 1)
    empty = np.flatnonzero(count == 0)
    if empty.size:
        rows, cols = to_pixel(compute_midpoints(mesh, mesh.topology.dim, empty.astype(np.int32))[:, :2], meta)
        mean[empty] = map_coordinates(image, [rows, cols], order=0, mode="nearest")
    field.x.array[space.dofmap.list[:n, 0]] = mean
    return field


def cell_values(field: fem.Function) -> np.ndarray:
    """A one-value-per-triangle function in cell order."""
    n = field.function_space.mesh.topology.index_map(field.function_space.mesh.topology.dim).size_local
    return field.x.array[field.function_space.dofmap.list[:n, 0]]


def cell_areas(mesh) -> np.ndarray:
    """Each triangle's area in um2, in cell order."""
    space = fem.functionspace(mesh, ("DG", 0))
    vector = fem.assemble_vector(fem.form(ufl.TestFunction(space) * ufl.dx)).array
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    return vector[space.dofmap.list[:n, 0]]


def mean_over(field: fem.Function) -> float:
    """A function's mean over the mesh: its integral over the area."""
    mesh = field.function_space.mesh
    one = fem.Constant(mesh, 1.0)
    return float(fem.assemble_scalar(fem.form(field * ufl.dx)) / fem.assemble_scalar(fem.form(one * ufl.dx)))


def steps(position: np.ndarray) -> np.ndarray:
    """The CYP2E1 gradient step, 0 to STEPS - 1, of a position."""
    return np.minimum(np.floor(position * STEPS / POSITION_SCALE), STEPS - 1).astype(np.int8)


def losses(back: np.ndarray, original: np.ndarray) -> dict:
    """Round trip against the original, as shares of the low-to-high range."""
    error = 100 * np.abs(back - original) / POSITION_SCALE
    return {"mean_error_pct": round(float(error.mean()), 3), "p95_error_pct": round(float(np.percentile(error, 95)), 3),
            "step_changed_pct": round(100 * float((steps(back) != steps(original)).mean()), 3)}


def round_trip(mesh_path: Path, layers: dict, meta: dict) -> tuple[dict, dict]:
    """One mesh: CYP2E1 at the corners and per triangle, necrosis per triangle, all back on the pixels; checks, arrays."""
    start = time.time()
    mesh, areas = read_mesh(mesh_path)
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    domain = layers["tissue"] & ~layers["lumen"]
    rows, cols = np.nonzero(domain)
    um, (oy, ox) = meta["um_per_px"], meta["origin_um"]
    points = np.column_stack([ox + cols * um, oy + rows * um, np.zeros(len(rows))])
    cells = cells_of(mesh, points)
    inside = cells >= 0

    image, cyp_measured = filled_cyp2e1(layers)
    corners = at_corners(mesh, image, meta)
    triangles = per_triangle(mesh, cells, image[rows, cols], image, meta)
    necrotic_pixels = (layers["necrotic"] & layers["measured"]).astype(np.float64)
    share = cell_values(per_triangle(mesh, cells, necrotic_pixels[rows, cols], necrotic_pixels, meta))
    necrotic = share >= NECROTIC_SHARE

    back_corners = np.full(domain.shape, np.nan)
    back_corners[rows[inside], cols[inside]] = corners.eval(points[inside], cells[inside]).ravel()
    back_triangles = np.full(domain.shape, np.nan)
    back_triangles[rows[inside], cols[inside]] = cell_values(triangles)[cells[inside]]
    back_necrotic = np.full(domain.shape, NO_VALUE, dtype=np.uint8)
    back_necrotic[rows[inside], cols[inside]] = necrotic[cells[inside]]

    judged = domain & cyp_measured & ~np.isnan(back_corners)
    original = layers["cyp2e1"][judged].astype(np.float64)
    area = cell_areas(mesh)
    tags = np.zeros(n, dtype=np.int32)
    tags[areas.indices] = areas.values
    in_mesh = domain & (back_necrotic != NO_VALUE)
    checks = {
        "triangles": n,
        "corners": int(corners.function_space.dofmap.index_map.size_local),
        "seconds": None,
        "pixels_outside_mesh_pct": round(100 * float((~inside).mean()), 3),
        "triangles_without_pixel_pct": round(100 * float((np.bincount(cells[inside], minlength=n) == 0).mean()), 3),
        "cyp2e1_gap_filled_pct": round(100 * float((domain & ~cyp_measured).sum() / domain.sum()), 2),
        "cyp2e1_corners": losses(back_corners[judged], original),
        "cyp2e1_triangles": losses(back_triangles[judged], original),
        "cyp2e1_mean": {"pixels": round(float(image[domain].mean()), 3), "corners": round(mean_over(corners), 3),
                        "triangles": round(mean_over(triangles), 3)},
        "necrotic_pct": {"pixels": round(100 * float(necrotic_pixels[domain].mean()), 3),
                         "triangles": round(100 * float(area[necrotic].sum() / area.sum()), 3)},
        "necrosis_pixels_changed_pct": round(100 * float((back_necrotic[in_mesh] != necrotic_pixels[in_mesh]).mean()), 3),
    }
    if AREA_TAGS["necrotic"] in tags:
        tagged = tags == AREA_TAGS["necrotic"]
        checks["necrotic_pct"]["area_names"] = round(100 * float(area[tagged].sum() / area.sum()), 3)
        checks["names_agree_with_share_pct"] = round(100 * float(area[tagged == necrotic].sum() / area.sum()), 3)
    checks["seconds"] = round(time.time() - start, 2)
    arrays = {
        "corner_xy_um": corners.function_space.tabulate_dof_coordinates()[:, :2],
        "cyp2e1_corners": corners.x.array.copy(),
        "triangle_midpoints_um": compute_midpoints(mesh, mesh.topology.dim, np.arange(n, dtype=np.int32))[:, :2],
        "cyp2e1_triangles": cell_values(triangles),
        "necrotic_share": share,
        "necrotic": necrotic,
        "cyp2e1_back_corners": back_corners.astype(np.float32),
        "cyp2e1_back_triangles": back_triangles.astype(np.float32),
        "necrotic_back": back_necrotic,
    }
    return checks, arrays


def main() -> None:
    """Round-trip every mesh in the folder, write its arrays and record, and a summary over all meshes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=Path, required=True, help="a build_mesh.py output folder")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    for record_path in sorted(args.meshes.glob("*.json")):
        mesh_record = json.loads(record_path.read_text())
        polygons_meta = json.loads(Path(mesh_record["polygons"]["file"]).read_text())["meta"]
        inputs_path = Path(polygons_meta["inputs"]["file"])
        layers, meta = read_inputs(inputs_path)
        mesh_path = record_path.with_suffix(".xdmf")
        checks, arrays = round_trip(mesh_path, layers, meta)
        record = {
            "slide": meta["slide"],
            "animal": meta["animal"],
            "mesh": {"file": str(mesh_path), "sha256": sha256(mesh_path.with_suffix(".h5")),
                     "variant": mesh_record["variant"], "size_um": mesh_record["size_um"], "code": mesh_record["code"]},
            "inputs": {"file": str(inputs_path), "sha256": sha256(inputs_path)},
            "rules": {"cyp2e1_gaps": "each pixel without a value takes the nearest measured value",
                      "corners": "Lagrange degree 1, each corner read between its four nearest pixels",
                      "triangles": "DG degree 0, mean of the pixel centres inside, the pixel at the middle if none",
                      "necrotic": f"necrotic share of the pixels at least {NECROTIC_SHARE:g}",
                      "errors": "pixels with a measured CYP2E1 value only, as % of the low-to-high range"},
            "checks": checks,
            "code": code_version(),
            "written": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        stem = record_path.stem
        np.savez_compressed(args.out / f"{stem}_fields.npz", **arrays)
        (args.out / f"{stem}_fields.json").write_text(json.dumps(record, indent=1))
        summary.append({"mesh": stem, "variant": mesh_record["variant"], **mesh_record["size_um"], **checks})
        print(f"{stem}: {json.dumps(checks)}", flush=True)
    (args.out / "fields_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
