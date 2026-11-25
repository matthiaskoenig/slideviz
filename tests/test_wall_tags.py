"""Portal and central wall tags of a mesh, from a lumen class raster. Runs in the mesh pixi env (gmsh, DOLFINx)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import build_mesh

BOXES = {"portal": (20, 20, 40, 40), "central": (60, 20, 80, 40), "plain": (20, 60, 40, 80)}  # x0, y0, x1, y1


def edges_in(view: dict, bounds: tuple) -> set[int]:
    """The edge tags of the mesh's edge segments lying within a box."""
    x0, y0, x1, y1 = bounds
    p = view["points_um"][view["edges"]]
    inside = ((p[..., 0] >= x0 - 1e-6) & (p[..., 0] <= x1 + 1e-6)
              & (p[..., 1] >= y0 - 1e-6) & (p[..., 1] <= y1 + 1e-6)).all(axis=1)
    return {int(t) for t in view["edge_tag"][inside]}


def test_wall_tags(tmp_path: Path) -> None:
    domain = box(0, 0, 100, 100)
    for x0, y0, x1, y1 in BOXES.values():
        domain = domain.difference(box(x0, y0, x1, y1))
    lumen = np.zeros((100, 100), bool)
    lumen_class = np.zeros((100, 100), np.uint8)
    for x0, y0, x1, y1 in BOXES.values():
        lumen[y0:y1, x0:x1] = True
    lumen_class[20:40, 20:40] = 1
    lumen_class[20:40, 60:80] = 2
    lumen_class[60:80, 20:40] = 3  # classified neither portal nor central
    meta = {"um_per_px": 1.0, "origin_um": [0.0, 0.0], "animal": "t", "slide": "t"}
    # a name with dots, as a dotted slide stem or a decimal size gives: the files keep it whole
    base = tmp_path / "t.v2_plain_h4-10.5um"
    checks = build_mesh.build("plain", [("tissue", domain)], lumen, lumen_class, meta, 4.0, 10.5, base)
    for suffix in (".msh", ".xdmf", ".h5", "_view.npz"):
        assert (tmp_path / f"t.v2_plain_h4-10.5um{suffix}").exists(), suffix
    with np.load(tmp_path / "t.v2_plain_h4-10.5um_view.npz") as view:
        view = dict(view)
    assert edges_in(view, BOXES["portal"]) >= {15}
    assert 16 not in edges_in(view, BOXES["portal"]) and 11 not in edges_in(view, BOXES["portal"])
    assert 16 in edges_in(view, BOXES["central"]) and 15 not in edges_in(view, BOXES["central"])
    assert edges_in(view, BOXES["plain"]) >= {11}
    assert not edges_in(view, BOXES["plain"]) & {15, 16}
    assert "portal_wall" in checks["dolfinx_names"] and "central_wall" in checks["dolfinx_names"]
