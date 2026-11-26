"""The wall class check: lumina by class, and those inside or near the necrosis, on synthetic rasters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("slideviz")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wall_class_check

UM = 2.0
SHAPE = (200, 300)
CENTRE = (100, 100)  # the necrotic disc, working px
RADIUS_PX = 20  # 40 um
PORTAL, CENTRAL, OTHER = 1, 2, 3
# lumina as 3 x 3 px squares: (class, row, col of the square's corner nearest the disc); the gap to the disc's edge
# along the row in um is (col - (CENTRE[1] + RADIUS_PX)) * UM
LUMINA = [
    (PORTAL, 99, 95),  # inside the disc
    (CENTRAL, 99, 115),  # 2 px into the disc's rim: inside
    (CENTRAL, 99, 135),  # 15 px = 30 um from the edge
    (OTHER, 99, 142),  # 22 px = 44 um
    (PORTAL, 99, 150),  # 30 px = 60 um: not near
    (CENTRAL, 99, 190),  # 70 px = 140 um
    (PORTAL, 20, 250),  # far
    (OTHER, 180, 250),  # far
]


def rasters() -> tuple[np.ndarray, np.ndarray]:
    lumen_class = np.zeros(SHAPE, np.uint8)
    for cls, r, c in LUMINA:
        lumen_class[r:r + 3, c:c + 3] = cls
    rows, cols = np.indices(SHAPE)
    necrotic = (rows - CENTRE[0]) ** 2 + (cols - CENTRE[1]) ** 2 <= RADIUS_PX**2
    return lumen_class, necrotic & (lumen_class == 0)  # a lumen is never necrotic tissue itself


def test_counts_by_class_near_necrosis() -> None:
    lumen_class, necrotic = rasters()
    counts = wall_class_check.count_lumina(lumen_class, necrotic, UM, radius_um=50.0)
    assert counts["lumina"] == {"portal": 3, "central": 3, "other": 2}
    assert counts["near_necrosis"] == {"portal": 1, "central": 2, "other": 1}
    assert counts["portal_central_ratio"] == pytest.approx(1.0)
    assert counts["near_portal_share"] == pytest.approx(1 / 4)


def test_radius_bounds_the_near_lumina() -> None:
    lumen_class, necrotic = rasters()
    counts = wall_class_check.count_lumina(lumen_class, necrotic, UM, radius_um=35.0)
    assert counts["near_necrosis"] == {"portal": 1, "central": 2, "other": 0}


def test_no_necrosis_no_near_lumina() -> None:
    lumen_class, necrotic = rasters()
    counts = wall_class_check.count_lumina(lumen_class, np.zeros_like(necrotic), UM)
    assert counts["near_necrosis"] == {"portal": 0, "central": 0, "other": 0}
    assert counts["near_portal_share"] is None


def test_gate_against_controls() -> None:
    def row(animal: str, portal: int, central: int, near_portal: int, near: int) -> dict:
        return {"animal": animal, "portal_central_ratio": portal / central,
                "near_portal_share": near_portal / near if near else None}

    rows = [row("000mg_m1", 20, 28, 0, 0), row("000mg_m2", 30, 30, 0, 0), row("281mg_m1", 25, 30, 1, 20),
            row("500mg_m1", 40, 20, 1, 20), row("375mg_m1", 25, 30, 3, 20)]
    gate = wall_class_check.gate(rows)
    assert gate["controls"] == ["000mg_m1", "000mg_m2"]
    assert gate["control_ratio_range"] == pytest.approx([20 / 28, 1.0])
    verdict = {s["animal"]: s for s in gate["dosed"]}
    assert verdict["281mg_m1"]["holds"] and verdict["281mg_m1"]["ratio_in_range"]
    assert not verdict["500mg_m1"]["ratio_in_range"] and not verdict["500mg_m1"]["holds"]
    assert not verdict["375mg_m1"]["near_portal_ok"] and not verdict["375mg_m1"]["holds"]
    assert not gate["holds"]
    json.dumps(gate)  # plain JSON types only
