"""Write a slidesim scene: scene.json for a mesh and its zonation position as a Zarr map.

Reads the mesh record of mesh/build_mesh.py (with its _view.npz and .msh beside it) and the <stem>_lobules.npz of
zonation_layers.py. The map is the position (0 periportal, 1 pericentral) cropped to the mesh bounding box plus a
margin, the NaN pixels filled by the nearest valid one, in the scene's [x, y] um frame (x = column, y = row). Runs
in the uv env; it needs neither gmsh nor DOLFINx.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jsonschema
import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt
from zarr.codecs import ZstdCodec

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "scene-1.schema.json"
MARGIN_PX = 2
CHUNK = 256
MAP_ID = "zonation"
MAP_PATH = f"maps/{MAP_ID}.zarr"


def fill_nearest(values: np.ndarray) -> np.ndarray:
    """`values` with each NaN replaced by the nearest finite value."""
    missing = np.isnan(values)
    if missing.all():
        raise ValueError("no position inside the mesh window")
    if not missing.any():
        return values
    rows, cols = distance_transform_edt(missing, return_distances=False, return_indices=True)
    return values[rows, cols]


def window(points_um: np.ndarray, um: float, origin_xy: np.ndarray, shape: tuple[int, int]) -> tuple[slice, slice]:
    """Rows and columns covering the points plus the margin, within the raster."""
    low = np.floor((points_um.min(axis=0) - origin_xy) / um).astype(int) - MARGIN_PX
    high = np.ceil((points_um.max(axis=0) - origin_xy) / um).astype(int) + MARGIN_PX + 1
    x0, y0 = np.maximum(low, 0)
    x1, y1 = np.minimum(high, [shape[1], shape[0]])
    if x0 >= x1 or y0 >= y1:
        raise ValueError("the mesh lies outside the position raster")
    return slice(int(y0), int(y1)), slice(int(x0), int(x1))


def write(record_path: Path, lobules: Path, out: Path) -> dict:
    """Write out/scene.json and out/maps/zonation.zarr; the scene document, validated against scene/1."""
    mesh_path = record_path.with_suffix(".msh")
    view_path = record_path.with_name(f"{record_path.stem}_view.npz")
    inputs = ((record_path, "mesh record", "mesh/build_mesh.py"), (mesh_path, "mesh .msh", "mesh/build_mesh.py"),
              (view_path, "mesh _view.npz", "mesh/build_mesh.py"), (lobules, "lobules file", "scripts/zonation_layers.py"))
    for path, what, builder in inputs:
        if not path.exists():
            raise FileNotFoundError(f"the {what} is missing: {path}; (re)build it with {builder}")
    edge_tags = json.loads(record_path.read_text())["tags"]["edges"]
    if not {"portal_wall", "central_wall"} <= edge_tags.keys():
        raise ValueError(f"{record_path} has no portal and central walls; rebuild the mesh with build_mesh.py")
    with np.load(view_path) as view:
        points_um = view["points_um"][:, :2]
    with np.load(lobules) as stored:
        position = stored["position"]
        um = float(stored["working_um_per_px"])
        origin_xy = np.asarray(stored["working_origin_um"], float)[::-1]  # [row, col] to [x, y]

    rows, cols = window(points_um, um, origin_xy, position.shape)
    cropped = fill_nearest(position[rows, cols].astype(np.float32))
    origin_um = [float(origin_xy[0] + cols.start * um), float(origin_xy[1] + rows.start * um)]

    out.mkdir(parents=True, exist_ok=True)
    target = out / MAP_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    array = zarr.create_array(store=str(target), shape=cropped.shape, chunks=(CHUNK, CHUNK), dtype=np.float32,
                              compressors=[ZstdCodec(level=3)], zarr_format=3, overwrite=True)
    array[:] = cropped

    scene = {
        "version": "scene/1",
        "dimension": 2,
        "units": {"length": "um"},
        "mesh": {"format": "gmsh", "path": Path(os.path.relpath(mesh_path, out)).as_posix()},
        "maps": [{"id": MAP_ID, "path": MAP_PATH, "units": "1", "um_per_px": um, "origin_um": origin_um}],
    }
    jsonschema.validate(scene, json.loads(SCHEMA.read_text()))
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    return scene


def main() -> None:
    """Write the scene of one mesh."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", type=Path, required=True, help="the .json record of a build_mesh.py mesh")
    parser.add_argument("--lobules", type=Path, required=True, help="<stem>_lobules.npz of zonation_layers.py")
    parser.add_argument("--out", type=Path, required=True, help="the scene folder")
    args = parser.parse_args()
    scene = write(args.mesh, args.lobules, args.out)
    shape = zarr.open_array(args.out / MAP_PATH, mode="r").shape
    print(f"{args.out / 'scene.json'}: {MAP_ID} {shape} at {scene['maps'][0]['origin_um']} um")


if __name__ == "__main__":
    main()
