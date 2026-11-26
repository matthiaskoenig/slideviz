"""The scene writer on a synthetic mesh view: schema, map frame and the nearest-neighbour fill (numpy only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import write_scene

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "scene-1.schema.json").read_text())
UM = 2.0
ORIGIN_RC = (-30.0, -50.0)  # working_origin_um [row, col]: pixel (0, 0) centred at x = -50, y = -30
SIZE = 100.0  # the mesh covers x, y in [0, 100] um


def square(points: list, x0: float, y0: float, x1: float, y1: float, tag: int, edges: list, tags: list) -> None:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    base = len(points)
    points.extend(corners)
    for i in range(4):
        edges.append((base + i, base + (i + 1) % 4))
        tags.append(tag)


def make_mesh(tmp_path: Path, stem: str = "m", listed: bool = True) -> Path:
    """A 100 um square with a portal hole at x 10-20 and a central hole at x 80-90, and its record <stem>.json, which
    lists its files (or not: listed False)."""
    points: list = [(0, 0), (SIZE, 0), (SIZE, SIZE), (0, SIZE)]
    edges: list = []
    tags: list = []
    square(points, 10, 40, 20, 60, 15, edges, tags)
    square(points, 80, 40, 90, 60, 16, edges, tags)
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    np.savez_compressed(mesh / f"{stem}_view.npz", points_um=np.array(points, float), triangles=np.array([[0, 1, 2]]),
                        edges=np.array(edges), edge_tag=np.array(tags))
    (mesh / f"{stem}.msh").write_text("$MeshFormat\n")
    record = {"tags": {"areas": {"tissue": 1}, "edges": {"vessel_wall": 11, "portal_wall": 15, "central_wall": 16}}}
    if listed:
        record["files"] = [f"{stem}{s}" for s in (".xdmf", ".h5", ".msh", "_view.npz")]
    (mesh / f"{stem}.json").write_text(json.dumps(record))
    return mesh / f"{stem}.json"


def make_lobules(tmp_path: Path) -> Path:
    """Position p = x / 100 on a raster larger than the mesh, NaN in the holes and outside the section."""
    rows, cols = 90, 110
    x = ORIGIN_RC[1] + np.arange(cols) * UM
    y = ORIGIN_RC[0] + np.arange(rows) * UM
    position = np.broadcast_to((x / SIZE)[None, :], (rows, cols)).astype(np.float32).copy()
    X, Y = np.meshgrid(x, y)
    position[(X < 0) | (X > SIZE) | (Y < 0) | (Y > SIZE)] = np.nan
    position[(X > 9) & (X < 21) & (Y > 39) & (Y < 61)] = np.nan
    position[(X > 79) & (X < 91) & (Y > 39) & (Y < 61)] = np.nan
    path = tmp_path / "m_lobules.npz"
    np.savez_compressed(path, position=position, lumen_class=np.zeros((rows, cols), np.uint8),
                        working_um_per_px=UM, working_origin_um=np.array(ORIGIN_RC))
    return path


@pytest.fixture
def scene(tmp_path: Path) -> tuple[Path, dict]:
    out = tmp_path / "out"
    write_scene.write(make_mesh(tmp_path), make_lobules(tmp_path), out)
    return out, json.loads((out / "scene.json").read_text())


def sample(array: np.ndarray, spec: dict, x: float, y: float) -> float:
    """Bilinear sample at (x, y) um, as the Map::sample of slidesim."""
    fx = (x - spec["origin_um"][0]) / spec["um_per_px"]
    fy = (y - spec["origin_um"][1]) / spec["um_per_px"]
    c0, r0 = int(np.floor(fx)), int(np.floor(fy))
    tx, ty = fx - c0, fy - r0
    a = array[r0:r0 + 2, c0:c0 + 2].astype(float)
    return float((1 - ty) * ((1 - tx) * a[0, 0] + tx * a[0, 1]) + ty * ((1 - tx) * a[1, 0] + tx * a[1, 1]))


def edge_midpoints(tmp_path: Path, tag: int) -> np.ndarray:
    with np.load(tmp_path / "mesh" / "m_view.npz") as view:
        p = view["points_um"][view["edges"][view["edge_tag"] == tag]]
    return p.mean(axis=1)


def test_scene_valid(scene: tuple[Path, dict]) -> None:
    out, doc = scene
    jsonschema.validate(doc, SCHEMA)
    assert doc["version"] == "scene/1" and doc["dimension"] == 2
    assert doc["mesh"] == {"format": "gmsh", "path": "../mesh/m.msh"}
    (spec,) = doc["maps"]
    assert spec["id"] == "zonation" and spec["path"] == "maps/zonation.zarr" and spec["um_per_px"] == UM
    array = zarr.open_array(out / spec["path"], mode="r")
    # the mesh box [0, 100] plus 2 px each side: 50 + 4 + 1 pixels per axis, within the raster
    assert array.shape == (55, 55) and array.dtype == np.float32
    assert spec["origin_um"] == [-4.0, -4.0]  # [x, y], 2 px before x = 0 and y = 0


def test_frame_matches_walls(tmp_path: Path, scene: tuple[Path, dict]) -> None:
    out, doc = scene
    (spec,) = doc["maps"]
    array = zarr.open_array(out / spec["path"], mode="r")[:]
    portal = [sample(array, spec, *m) for m in edge_midpoints(tmp_path, 15)]
    central = [sample(array, spec, *m) for m in edge_midpoints(tmp_path, 16)]
    assert max(portal) < 0.3 and min(central) > 0.7


def test_no_nan_in_window(scene: tuple[Path, dict]) -> None:
    out, doc = scene
    array = zarr.open_array(out / doc["maps"][0]["path"], mode="r")[:]
    assert not np.isnan(array).any()
    assert 0.0 <= array.min() and array.max() <= 1.0 + 1e-6


def test_window_without_section_fails(tmp_path: Path) -> None:
    lobules = make_lobules(tmp_path)
    with np.load(lobules) as stored:
        data = dict(stored)
    data["position"][:] = np.nan
    np.savez_compressed(lobules, **data)
    with pytest.raises(ValueError, match="no position"):
        write_scene.write(make_mesh(tmp_path), lobules, tmp_path / "out")


def test_missing_input_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="lobules file is missing.*zonation_layers.py"):
        write_scene.write(make_mesh(tmp_path), tmp_path / "none_lobules.npz", tmp_path / "out")


@pytest.mark.parametrize("listed", [True, False])
def test_dotted_stem(tmp_path: Path, listed: bool) -> None:
    """A mesh name with dots (a dotted slide stem, a decimal size): the .msh and _view.npz beside its record."""
    stem = "mouse.v2_plain_h2.5-10um"
    doc = write_scene.write(make_mesh(tmp_path, stem, listed), make_lobules(tmp_path), tmp_path / "out")
    assert doc["mesh"]["path"] == f"../mesh/{stem}.msh"
    assert doc["maps"][0]["origin_um"] == [-4.0, -4.0]  # the view of this mesh was read


def test_missing_view_named(tmp_path: Path) -> None:
    record = make_mesh(tmp_path, "mouse.v2_plain_h5-20um")
    (record.parent / "mouse.v2_plain_h5-20um_view.npz").unlink()
    with pytest.raises(FileNotFoundError, match=r"_view.npz is missing: .*mouse\.v2_plain_h5-20um_view\.npz"):
        write_scene.write(record, make_lobules(tmp_path), tmp_path / "out")
