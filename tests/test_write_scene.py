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
SHAPE = (90, 110)  # the working grid of make_lobules
HE = "mouse_x_he"


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
    rows, cols = SHAPE
    x = ORIGIN_RC[1] + np.arange(cols) * UM
    y = ORIGIN_RC[0] + np.arange(rows) * UM
    position = np.broadcast_to((x / SIZE)[None, :], (rows, cols)).astype(np.float32).copy()
    X, Y = np.meshgrid(x, y)
    position[(X < 0) | (X > SIZE) | (Y < 0) | (Y > SIZE)] = np.nan
    position[(X > 9) & (X < 21) & (Y > 39) & (Y < 61)] = np.nan
    position[(X > 79) & (X < 91) & (Y > 39) & (Y < 61)] = np.nan
    path = tmp_path / "m_lobules.npz"
    np.savez_compressed(path, position=position, lumen_class=np.zeros((rows, cols), np.uint8),
                        working_um_per_px=UM, working_origin_um=np.array(ORIGIN_RC), reference=HE)
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


def working_xy() -> tuple[np.ndarray, np.ndarray]:
    """x and y in um of the working pixel centres of make_lobules, each of shape SHAPE."""
    x = ORIGIN_RC[1] + np.arange(SHAPE[1]) * UM
    y = ORIGIN_RC[0] + np.arange(SHAPE[0]) * UM
    return np.meshgrid(x, y)


def make_readout(tmp_path: Path, position: np.ndarray) -> Path:
    path = tmp_path / "x_cyp2e1_readout.npz"
    np.savez_compressed(path, position=position.astype(np.uint8), um_per_px=UM, reference=HE)
    return path


def make_calls(tmp_path: Path, calls: np.ndarray, um: float) -> Path:
    """A call map whose pixel (0, 0) is centred on working pixel (0, 0) of make_lobules: the calls' level grid has
    pixel p centred at p * um in the H&E frame, so the map starts at ORIGIN_RC / um."""
    origin_level_rc = [round(ORIGIN_RC[0] / um), round(ORIGIN_RC[1] / um)]
    meta = {"animal": "x", "slide": HE, "level": 1, "um_per_px": um, "origin_level_rc": origin_level_rc}
    path = tmp_path / "x_necrosis.npz"
    np.savez_compressed(path, call_E=calls.astype(np.uint8), call_D=calls.astype(np.uint8), meta=json.dumps(meta))
    return path


def maps_of(out: Path, doc: dict) -> dict:
    return {spec["id"]: (spec, zarr.open_array(out / spec["path"], mode="r")) for spec in doc["maps"]}


def test_cell_maps_share_the_zonation_grid(tmp_path: Path) -> None:
    X, Y = working_xy()
    readout = np.where(X < 50, 125, 250)
    readout[(Y > 39) & (Y < 61) & (X > 9) & (X < 21)] = 255  # the portal hole: not tissue
    hole = (25, 60)  # a single pixel without readout among 250s; x = -50 + 2 * 60 = 70 um, y = -30 + 2 * 25 = 20 um
    readout[hole] = 255
    calls = ((X > 60) & (X < 90) & (Y > 10) & (Y < 30)).astype(np.uint8)
    calls[(Y > 39) & (Y < 61) & (X > 79) & (X < 91)] = 255  # the central hole: not measured
    unmeasured = (15, 70)  # x = 90 um, y = 0 um, outside the necrotic block, among 0s
    calls[unmeasured] = 255
    calls[23, 62] = 255  # x = 74 um, y = 16 um: a pixel inside the necrotic block, among 1s
    out = tmp_path / "out"
    doc = write_scene.write(make_mesh(tmp_path), make_lobules(tmp_path), out,
                            cyp2e1=make_readout(tmp_path, readout), necrosis=make_calls(tmp_path, calls, UM))
    jsonschema.validate(doc, SCHEMA)
    maps = maps_of(out, doc)
    assert sorted(maps) == ["cyp2e1", "necrosis", "zonation"]
    zonation_spec, zonation = maps["zonation"]
    for name in ("cyp2e1", "necrosis"):
        spec, array = maps[name]
        assert spec["units"] == "1" and spec["path"] == f"maps/{name}.zarr"
        assert array.shape == zonation.shape and array.dtype == np.float32
        assert spec["um_per_px"] == zonation_spec["um_per_px"] and spec["origin_um"] == zonation_spec["origin_um"]
    # the crop starts 2 px before x = 0 and y = 0: working pixel (r, c) is crop pixel (r - 13, c - 23)
    cyp2e1 = maps["cyp2e1"][1][:]
    assert cyp2e1[25 - 13, 30 - 23] == pytest.approx(0.5)  # x = 10 um, readout 125
    assert cyp2e1[hole[0] - 13, hole[1] - 23] == pytest.approx(1.0)  # its neighbours are 250
    assert cyp2e1[25 - 13, 70 - 23] == pytest.approx(1.0)  # x = 90 um, readout 250
    necrosis = maps["necrosis"][1][:]
    assert set(np.unique(necrosis)) == {0.0, 1.0}
    assert necrosis[25 - 13, 65 - 23] == 1.0  # x = 80 um, y = 20 um: call 1
    assert necrosis[23 - 13, 62 - 23] == 1.0  # not measured, among 1s
    assert necrosis[unmeasured[0] - 13, unmeasured[1] - 23] == 0.0  # not measured, among 0s
    assert necrosis[50 - 13, 30 - 23] == 0.0  # x = 10 um, y = 70 um: call 0
    # the filled share: the map's pixels without a value of their own, in the Zarr attributes of the map
    crop = (slice(13, 13 + 55), slice(23, 23 + 55))
    for name, missing in (("cyp2e1", readout == 255), ("necrosis", calls == 255)):
        provenance = maps[name][1].attrs["provenance"]
        assert provenance["filled_share"] == pytest.approx(float(missing[crop].mean()))
        assert provenance["fill"] == "nearest"


def test_necrosis_resampled_from_level_grid(tmp_path: Path) -> None:
    """Calls at a quarter of the working pixel: each working pixel is the majority of its 4 x 4 calls."""
    fine = UM / 4
    x = ORIGIN_RC[1] + np.arange(SHAPE[1] * 4) * fine
    y = ORIGIN_RC[0] + np.arange(SHAPE[0] * 4) * fine
    X, Y = np.meshgrid(x, y)
    block = (X >= 20) & (X < 60) & (Y >= 30) & (Y < 70)
    out = tmp_path / "out"
    doc = write_scene.write(make_mesh(tmp_path), make_lobules(tmp_path), out,
                            necrosis=make_calls(tmp_path, block, fine))
    spec, array = maps_of(out, doc)["necrosis"]
    necrosis = array[:]
    xs = spec["origin_um"][0] + np.arange(necrosis.shape[1]) * spec["um_per_px"]
    ys = spec["origin_um"][1] + np.arange(necrosis.shape[0]) * spec["um_per_px"]
    XS, YS = np.meshgrid(xs, ys)
    # the block's edges excepted: one working pixel on either side of them
    inside = (XS > 20 + UM) & (XS < 60 - UM) & (YS > 30 + UM) & (YS < 70 - UM)
    outside = (XS < 20 - UM) | (XS > 60 + UM) | (YS < 30 - UM) | (YS > 70 + UM)
    assert inside.sum() > 200 and outside.sum() > 1000
    assert (necrosis[inside] == 1.0).all() and (necrosis[outside] == 0.0).all()
    assert maps_of(out, doc)["necrosis"][1].attrs["provenance"]["filled_share"] == 0.0


def test_cell_maps_of_another_slide_rejected(tmp_path: Path) -> None:
    X, _ = working_xy()
    readout = make_readout(tmp_path, np.full(X.shape, 125))
    with np.load(readout) as stored:
        data = dict(stored)
    data["reference"] = "mouse_y_he"
    np.savez_compressed(readout, **data)
    with pytest.raises(ValueError, match="cyp2e1 map belongs to mouse_y_he, not mouse_x_he"):
        write_scene.write(make_mesh(tmp_path), make_lobules(tmp_path), tmp_path / "out", cyp2e1=readout)


def test_readout_on_another_grid_rejected(tmp_path: Path) -> None:
    readout = make_readout(tmp_path, np.full((SHAPE[0] // 2, SHAPE[1] // 2), 125))
    with pytest.raises(ValueError, match="cyp2e1 map .* is not on the working grid"):
        write_scene.write(make_mesh(tmp_path), make_lobules(tmp_path), tmp_path / "out", cyp2e1=readout)


def make_polygons(tmp_path: Path, tags: tuple[str, ...]) -> Path:
    """A polygons/1 file over the 100 um square with a line feature per edge tag (geometry is not checked here)."""
    square = [[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE], [0, 0]]
    features = [{"type": "Feature", "properties": {"kind": "area", "tag": "surviving", "value": 1},
                 "geometry": {"type": "Polygon", "coordinates": [square]}}]
    features += [{"type": "Feature", "properties": {"kind": "edge", "tag": tag, "value": 11 + i},
                  "geometry": {"type": "LineString", "coordinates": [[0, 0], [SIZE, 0]]}} for i, tag in enumerate(tags)]
    path = tmp_path / "m_polygons_1um.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "version": "polygons/1", "features": features}))
    return path


def test_polygons_scene_validates(tmp_path: Path) -> None:
    tags = ("vessel_wall", "portal_wall", "central_wall", "section_edge", "necrosis_border", "field_cut")
    out = tmp_path / "out"
    doc = write_scene.write(None, make_lobules(tmp_path), out, polygons=make_polygons(tmp_path, tags), size_um="5:20")
    jsonschema.validate(doc, SCHEMA)
    jsonschema.validate(json.loads((tmp_path / "m_polygons_1um.geojson").read_text()),
                        json.loads((Path(__file__).resolve().parents[1] / "schemas" / "polygons-1.schema.json").read_text()))
    assert doc["mesh"] == {"format": "polygons", "path": "../m_polygons_1um.geojson",
                           "size": {"min_um": 5.0, "max_um": 20.0, "dist_min_um": 10.0, "dist_max_um": 200.0,
                                    "walls": ["vessel_wall", "portal_wall", "central_wall", "section_edge",
                                              "necrosis_border"]}}
    assert doc["maps"][0]["origin_um"] == [-4.0, -4.0]  # the window follows the polygons' bounding box


def test_polygons_scene_walls_are_those_present(tmp_path: Path) -> None:
    doc = write_scene.write(None, make_lobules(tmp_path), tmp_path / "out",
                            polygons=make_polygons(tmp_path, ("portal_wall", "field_cut")), size_um="2.5:10")
    assert doc["mesh"]["size"]["walls"] == ["portal_wall"] and doc["mesh"]["size"]["dist_min_um"] == 5.0


@pytest.mark.parametrize("kwargs", [{"size_um": "5:20"}, {"polygons": True}, {"polygons": True, "size_um": "20:5"},
                                    {"polygons": True, "size_um": "5"}])
def test_polygons_scene_arguments_rejected(tmp_path: Path, kwargs: dict) -> None:
    polygons = make_polygons(tmp_path, ("portal_wall",)) if kwargs.get("polygons") else None
    with pytest.raises(ValueError):
        write_scene.write(None, make_lobules(tmp_path), tmp_path / "out", polygons=polygons,
                          size_um=kwargs.get("size_um"))
