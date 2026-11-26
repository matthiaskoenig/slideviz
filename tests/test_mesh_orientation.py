"""Every triangle of a built mesh is counterclockwise in the [x, y] um frame, whatever the rings' orientation.

Runs in the mesh pixi env (gmsh, DOLFINx).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from shapely.geometry import box
from shapely.geometry.polygon import orient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import build_mesh

META = {"um_per_px": 1.0, "origin_um": [0.0, 0.0], "animal": "t", "slide": "t"}


def signed_areas(base: Path) -> np.ndarray:
    with np.load(base.with_name(base.name + "_view.npz")) as view:
        p, t = view["points_um"], view["triangles"]
    u, v = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]
    return 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])


def field(sign: float):
    """A 100 um square with a lumen, its outer ring counterclockwise (sign 1) or clockwise (-1), as polygons.py may
    give either."""
    return orient(box(0, 0, 100, 100).difference(box(40, 40, 60, 60)), sign=sign)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_plain_triangles_counterclockwise(tmp_path: Path, sign: float) -> None:
    lumen = np.zeros((100, 100), bool)
    lumen[40:60, 40:60] = True
    base = tmp_path / "t_plain"
    build_mesh.build("plain", [("tissue", field(sign))], lumen, np.zeros((100, 100), np.uint8), META, 5.0, 10.0, base)
    area = signed_areas(base)
    assert (area > 0).all(), f"{int((area <= 0).sum())} of {len(area)} triangles not counterclockwise"


def test_follow_triangles_counterclockwise(tmp_path: Path) -> None:
    """Two areas of opposite ring orientation, as surviving and necrotic polygons may come."""
    surviving = orient(box(0, 0, 50, 100), sign=-1.0)
    necrotic = orient(box(50, 0, 100, 100), sign=1.0)
    base = tmp_path / "t_follow"
    build_mesh.build("follow", [("surviving", surviving), ("necrotic", necrotic)], np.zeros((100, 100), bool),
                     np.zeros((100, 100), np.uint8), META, 5.0, 10.0, base)
    area = signed_areas(base)
    assert (area > 0).all(), f"{int((area <= 0).sum())} of {len(area)} triangles not counterclockwise"
