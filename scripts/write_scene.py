"""Write a slidesim scene: scene.json for a mesh and its zonation position, CYP2E1 and necrosis as Zarr maps.

Reads the mesh record of mesh/build_mesh.py (with its _view.npz and .msh beside it) and the <stem>_lobules.npz of
zonation_layers.py. The zonation map is the position (0 periportal, 1 pericentral) cropped to the mesh bounding box
plus a margin, in the scene's [x, y] um frame (x = column, y = row). Optional maps on the same grid and crop:

- cyp2e1: the position of cyp2e1_readout.py (on the H&E working grid) divided by its 250, the DAB between the low
  (0) and high (1) of its 500 um window;
- necrosis: the call_E map of necrosis_slides.py, 1 necrotic or 0, each working pixel by the majority of its calls.

Pixels without a value (outside the section, not tissue, not measured) take the nearest valid one; the share of such
pixels and the source of each map are in the provenance attribute of its Zarr array (scene/1 has no field for them).
Runs in the uv env; it needs neither gmsh nor DOLFINx.
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

from slideviz.analysis.dab import NO_TISSUE, POSITION_SCALE
from slideviz.analysis.prediction import calls_on_grid, read_calls
from slideviz.data.registration import sha256

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "scene-1.schema.json"
MARGIN_PX = 2
CHUNK = 256
MAP_ID = "zonation"
CALL_KEY = "call_E"  # the necrosis call of necrosis_slides.py the scene takes


def fill_nearest(values: np.ndarray, what: str = "position") -> np.ndarray:
    """`values` with each NaN replaced by the nearest finite value."""
    missing = np.isnan(values)
    if missing.all():
        raise ValueError(f"no {what} inside the mesh window")
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


def mesh_file(record_path: Path, record: dict, suffix: str) -> Path:
    """The file of a mesh record ending in suffix (".msh", "_view.npz"), beside the record.

    The record's "files" list names it; a record without the list gives its own name without ".json" plus the suffix.
    Never Path.with_suffix: mesh names have dots (mouse.v2_plain_h2.5-10um), and it would cut them at the first one.
    """
    named = [f for f in record.get("files", []) if f.endswith(suffix)]
    if len(named) > 1:
        raise ValueError(f"{record_path} lists several {suffix} files: {named}")
    name = named[0] if named else record_path.name.removesuffix(".json") + suffix
    return record_path.with_name(name)


def map_path(map_id: str) -> str:
    return f"maps/{map_id}.zarr"


def readout_values(path: Path, reference: str, um: float, shape: tuple[int, int], rows: slice,
                   cols: slice) -> np.ndarray:
    """The CYP2E1 readout over the window, 0 to 1, NaN outside the tissue."""
    with np.load(path) as stored:
        position, readout_um, slide = stored["position"], float(stored["um_per_px"]), str(stored["reference"])
    if slide != reference:
        raise ValueError(f"the cyp2e1 map belongs to {slide}, not {reference}")
    if position.shape != shape or not np.isclose(readout_um, um, rtol=1e-9, atol=0):
        raise ValueError(f"the cyp2e1 map {position.shape} at {readout_um} um/px is not on the working grid {shape} "
                         f"at {um} um/px of the lobules file")
    window = position[rows, cols]
    bad = (window > POSITION_SCALE) & (window != NO_TISSUE)
    if bad.any():
        raise ValueError(f"the cyp2e1 map has values {sorted(np.unique(window[bad]))[:5]} above {POSITION_SCALE} "
                         f"other than {NO_TISSUE}")
    return np.where(window == NO_TISSUE, np.nan, window / POSITION_SCALE).astype(np.float32)


def necrosis_values(path: Path, reference: str, um: float, origin_rc: np.ndarray, rows: slice,
                    cols: slice) -> tuple[np.ndarray, dict]:
    """The necrosis calls over the window, 1 necrotic, 0 not, NaN not measured; and the call map's record."""
    calls, meta = read_calls(path, CALL_KEY)
    if meta["slide"] != reference:
        raise ValueError(f"the necrosis map belongs to {meta['slide']}, not {reference}")
    necrotic, measured = calls_on_grid(calls, meta, um, tuple(origin_rc), rows, cols)
    return np.where(measured, necrotic, np.nan).astype(np.float32), meta


def save_map(out: Path, map_id: str, values: np.ndarray, provenance: dict) -> None:
    """The map as Zarr v3 at out/maps/<id>.zarr, its NaN filled by the nearest value; the provenance as attribute."""
    filled = fill_nearest(values, "position" if map_id == MAP_ID else f"{map_id} value")
    target = out / map_path(map_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = zarr.create_array(store=str(target), shape=filled.shape, chunks=(CHUNK, CHUNK), dtype=np.float32,
                              compressors=[ZstdCodec(level=3)], zarr_format=3, overwrite=True,
                              attributes={"provenance": {**provenance, "fill": "nearest",
                                                         "filled_share": float(np.isnan(values).mean())}})
    array[:] = filled


def source(path: Path) -> dict:
    return {"source": str(path.resolve()), "sha256": sha256(path)}


def write(record_path: Path, lobules: Path, out: Path, cyp2e1: Path | None = None,
          necrosis: Path | None = None) -> dict:
    """Write out/scene.json and out/maps/<id>.zarr; the scene document, validated against scene/1."""
    if not record_path.exists():
        raise FileNotFoundError(f"the mesh record is missing: {record_path}; (re)build it with mesh/build_mesh.py")
    record = json.loads(record_path.read_text())
    mesh_path = mesh_file(record_path, record, ".msh")
    view_path = mesh_file(record_path, record, "_view.npz")
    inputs = ((mesh_path, "mesh .msh", "mesh/build_mesh.py"), (view_path, "mesh _view.npz", "mesh/build_mesh.py"),
              (lobules, "lobules file", "scripts/zonation_layers.py"),
              *(((cyp2e1, "cyp2e1 readout", "scripts/cyp2e1_readout.py"),) if cyp2e1 else ()),
              *(((necrosis, "necrosis call map", "scripts/necrosis_slides.py"),) if necrosis else ()))
    for path, what, builder in inputs:
        if not path.exists():
            raise FileNotFoundError(f"the {what} is missing: {path}; (re)build it with {builder}")
    edge_tags = record["tags"]["edges"]
    if not {"portal_wall", "central_wall"} <= edge_tags.keys():
        raise ValueError(f"{record_path} has no portal and central walls; rebuild the mesh with build_mesh.py")
    with np.load(view_path) as view:
        points_um = view["points_um"][:, :2]
    with np.load(lobules) as stored:
        position = stored["position"]
        um = float(stored["working_um_per_px"])
        origin_rc = np.asarray(stored["working_origin_um"], float)
        reference = str(stored["reference"]) if "reference" in stored else None
    if (cyp2e1 or necrosis) and reference is None:
        raise ValueError(f"{lobules} names no H&E slide (reference); rebuild it with scripts/zonation_layers.py")
    origin_xy = origin_rc[::-1]  # [row, col] to [x, y]

    rows, cols = window(points_um, um, origin_xy, position.shape)
    origin_um = [float(origin_xy[0] + cols.start * um), float(origin_xy[1] + rows.start * um)]
    maps = {MAP_ID: (position[rows, cols].astype(np.float32),
                     {**source(lobules), "values": "zonation position, 0 periportal, 1 pericentral"})}
    if cyp2e1:
        maps["cyp2e1"] = (readout_values(cyp2e1, reference, um, position.shape, rows, cols),
                          {**source(cyp2e1), "values": f"cyp2e1_readout.py position / {POSITION_SCALE}: the DAB between "
                                                                "its window's low (0) and high (1)"})
    if necrosis:
        values, meta = necrosis_values(necrosis, reference, um, origin_rc, rows, cols)
        maps["necrosis"] = (values, {**source(necrosis), "values": f"{CALL_KEY}, 1 necrotic, 0 not",
                                     "cutoff": meta.get("cutoff"),
                                     "resampling": f"majority of the calls at {meta['um_per_px']} um/px per pixel"})

    out.mkdir(parents=True, exist_ok=True)
    for map_id, (values, provenance) in maps.items():
        save_map(out, map_id, values, provenance)

    scene = {
        "version": "scene/1",
        "dimension": 2,
        "units": {"length": "um"},
        "mesh": {"format": "gmsh", "path": Path(os.path.relpath(mesh_path, out)).as_posix()},
        "maps": [{"id": map_id, "path": map_path(map_id), "units": "1", "um_per_px": um, "origin_um": origin_um}
                 for map_id in maps],
    }
    jsonschema.validate(scene, json.loads(SCHEMA.read_text()))
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    return scene


def main() -> None:
    """Write the scene of one mesh."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", type=Path, required=True, help="the .json record of a build_mesh.py mesh")
    parser.add_argument("--lobules", type=Path, required=True, help="<stem>_lobules.npz of zonation_layers.py")
    parser.add_argument("--cyp2e1", type=Path, help="<stem>_cyp2e1_readout.npz of cyp2e1_readout.py, for a cyp2e1 map")
    parser.add_argument("--necrosis", type=Path, help="call map of necrosis_slides.py, for a necrosis map")
    parser.add_argument("--out", type=Path, required=True, help="the scene folder")
    args = parser.parse_args()
    scene = write(args.mesh, args.lobules, args.out, cyp2e1=args.cyp2e1, necrosis=args.necrosis)
    for spec in scene["maps"]:
        array = zarr.open_array(args.out / spec["path"], mode="r")
        filled = array.attrs["provenance"]["filled_share"]
        print(f"{args.out / 'scene.json'}: {spec['id']} {array.shape} at {spec['origin_um']} um, "
              f"{100 * filled:.1f} % filled")


if __name__ == "__main__":
    main()
