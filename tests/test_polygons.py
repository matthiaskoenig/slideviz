"""polygons.py end to end on a synthetic field: polygons/1 schema, exact shared coordinates, slidesim's own validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mesh"))
sys.path.insert(0, str(ROOT / "scripts"))
import polygons
from tags import AREA_TAGS, EDGE_TAGS

SCHEMA = json.loads((ROOT / "schemas" / "polygons-1.schema.json").read_text())
RESERVED = {"depth", "celltype", "Cell Sets", "Face Sets", "Edge Sets", "Vertex Sets", "marker", "ghost", "vtk",
            "_boundary_", "_regions_", "pde_floating"}
SLIDESIM = Path(os.environ.get("SLIDESIM_BIN", Path.home() / "git" / "slidesim" / "target" / "release" / "slidesim"))
N = 120


def write_inputs(path: Path, margin: int) -> None:
    """A 120 x 120 um field (1 um/px): tissue with `margin` px of outside, a portal lumen, a central lumen at the
    border of a necrotic patch, and a round tear."""
    tissue = np.zeros((N, N), bool)
    tissue[margin:N - margin, margin:N - margin] = True
    lumen = np.zeros((N, N), bool)
    lumen_class = np.zeros((N, N), np.uint8)
    lumen[20:36, 20:36] = True
    lumen_class[20:36, 20:36] = 1
    lumen[80:92, 20:32] = True
    lumen_class[80:92, 20:32] = 2
    yy, xx = np.mgrid[:N, :N]
    lumen |= (yy - 40) ** 2 + (xx - 90) ** 2 < 36
    necrotic = np.zeros((N, N), bool)
    necrotic[55:100, 60:105] = True
    meta = {"um_per_px": 1.0, "origin_um": [0.0, 0.0], "animal": "syn", "slide": "syn", "code": "test"}
    np.savez(path, tissue=tissue, lumen=lumen, necrotic=necrotic, lumen_class=lumen_class,
             measured=np.ones((N, N), bool), meta=json.dumps(meta))


def run_polygons(tmp_path: Path, variant: str, margin: int, tolerance: float = 1.0) -> dict:
    inputs = tmp_path / "inputs.npz"
    write_inputs(inputs, margin)
    out = tmp_path / f"out_{variant}_{margin}"
    argv = ["polygons.py", "--inputs", str(inputs), "--tolerance-um", str(tolerance), "--variant", variant,
            "--out", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        polygons.main()
    finally:
        sys.argv = old
    (path,) = out.glob("*.geojson")
    return {"path": path, "doc": json.loads(path.read_text())}


def segments(coords: list) -> set[tuple]:
    return {(min(a, b), max(a, b)) for a, b in pairwise([tuple(c) for c in coords])}


def rings(geometry: dict) -> list[list]:
    parts = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return [ring for part in parts for ring in part]


@pytest.mark.parametrize("variant", ["follow", "plain"])
@pytest.mark.parametrize("margin", [0, 6])
def test_polygons_1_is_exact(tmp_path: Path, variant: str, margin: int) -> None:
    doc = run_polygons(tmp_path, variant, margin)["doc"]
    jsonschema.validate(doc, SCHEMA)
    assert doc["version"] == "polygons/1"
    props = [f["properties"] for f in doc["features"]]
    assert {p["tag"] for p in props} <= (set(AREA_TAGS) | set(EDGE_TAGS)) - RESERVED
    assert all(p["value"] == (AREA_TAGS if p["kind"] == "area" else EDGE_TAGS)[p["tag"]] for p in props)
    area_tags = {p["tag"] for p in props if p["kind"] == "area"}
    assert area_tags == ({"surviving", "necrotic"} if variant == "follow" else {"tissue"})
    edge_tags = {p["tag"] for p in props if p["kind"] == "edge"}
    assert {"portal_wall", "central_wall", "vessel_wall"} <= edge_tags
    assert ("field_cut" in edge_tags) == (margin == 0) and ("section_edge" in edge_tags) == (margin > 0)
    assert ("necrosis_border" in edge_tags) == (variant == "follow")
    # every segment of the areas' rings is once in a chain, and every chain segment is one of them, bit for bit
    count: dict = {}
    for f in doc["features"]:
        if f["properties"]["kind"] == "area":
            for ring in rings(f["geometry"]):
                for s in segments(ring):
                    count[s] = count.get(s, 0) + 1
    chained = []
    for f in doc["features"]:
        if f["properties"]["kind"] == "edge":
            lines = [f["geometry"]["coordinates"]] if f["geometry"]["type"] == "LineString" else f["geometry"]["coordinates"]
            chained += [s for line in lines for s in segments(line)]
    assert sorted(chained) == sorted(count) and len(chained) == len(set(chained))


@pytest.mark.skipif(not SLIDESIM.exists(), reason="no slidesim binary (cargo build --release -p slidesim)")
@pytest.mark.parametrize("variant", ["follow", "plain"])
@pytest.mark.parametrize("margin", [0, 6])
def test_slidesim_accepts_the_polygons(tmp_path: Path, variant: str, margin: int) -> None:
    import write_scene

    made = run_polygons(tmp_path, variant, margin)
    lobules = tmp_path / "syn_lobules.npz"
    position = np.full((N, N), 0.5, np.float32)
    np.savez(lobules, position=position, lumen_class=np.zeros((N, N), np.uint8), working_um_per_px=1.0,
             working_origin_um=np.array([0.0, 0.0]), reference="syn")
    out = tmp_path / "scene"
    write_scene.write(made["path"], "5:20", lobules, out)
    result = subprocess.run([str(SLIDESIM), "mesh", str(out / "scene.json"), "--out", str(tmp_path / "mesh")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    quality = json.loads((tmp_path / "mesh" / "quality.json").read_text())
    assert quality["triangles"] > 100 and quality["min_angle_deg"] > 20, quality
