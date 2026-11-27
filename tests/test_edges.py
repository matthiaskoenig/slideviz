"""Boundary chains of tagged areas: each boundary segment once, the wall classes of the gmsh mesher's fixture."""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import edges

HOLES = {"portal": (20, 20, 40, 40), "central": (60, 20, 80, 40), "plain": (20, 60, 40, 80)}  # x0, y0, x1, y1


def segments(chains: dict) -> list[tuple]:
    """Undirected segments of all chains, in a canonical order."""
    return sorted((min(a, b), max(a, b)) for lines in chains.values() for line in lines for a, b in pairwise(line))


def rings(polygons: list) -> list[tuple]:
    """Undirected boundary segments of the polygons that belong to exactly one of them."""
    count: dict = {}
    for polygon in polygons:
        for a, b, _ in edges.ring_segments(polygon):
            count[(min(a, b), max(a, b))] = count.get((min(a, b), max(a, b)), 0) + 1
    return sorted(k for k in count)


def on(chains: dict, tag: str, bounds: tuple) -> bool:
    """All segments of a tag's chains lie within the box, and the tag has some."""
    x0, y0, x1, y1 = bounds
    points = np.array([p for line in chains.get(tag, []) for p in line])
    return bool(len(points)) and bool(((points >= [x0 - 1e-6, y0 - 1e-6]) & (points <= [x1 + 1e-6, y1 + 1e-6])).all())


def test_edges_cover_the_boundary() -> None:
    patch = box(60, 60, 80, 80)
    hole = box(20, 40, 40, 60)
    surviving = box(0, 0, 100, 100).difference(hole).difference(patch)
    lumen = np.zeros((100, 100), bool)
    lumen[40:60, 20:40] = True
    lumen_class = np.zeros((100, 100), np.uint8)
    lumen_class[40:60, 20:40] = 1
    meta = {"um_per_px": 1.0, "origin_um": [0.5, 0.5]}
    chains = edges.edge_chains([surviving, patch], edges.frame(meta, lumen.shape), lumen, lumen_class, meta)
    # each distinct boundary segment of the areas is in exactly one chain
    got = segments(chains)
    assert got == rings([surviving, patch]) and len(got) == len(set(got))
    assert set(chains) == {"portal_wall", "field_cut", "necrosis_border"}
    assert on(chains, "portal_wall", (20, 40, 40, 60)) and len(chains["portal_wall"]) == 1
    assert on(chains, "necrosis_border", (60, 60, 80, 80))
    assert len(chains["field_cut"]) == 1  # the four sides join to one closed chain
    assert chains["field_cut"][0][0] == chains["field_cut"][0][-1]


def test_section_edge_where_the_domain_ends_inside_the_field() -> None:
    tissue = box(10, 10, 90, 90)
    lumen = np.zeros((100, 100), bool)
    meta = {"um_per_px": 1.0, "origin_um": [0.5, 0.5]}
    chains = edges.edge_chains([tissue], edges.frame(meta, lumen.shape), lumen, np.zeros((100, 100), np.uint8), meta)
    assert set(chains) == {"section_edge"}


def test_wall_classes_match_build_mesh() -> None:
    """The fixture of test_wall_tags.py: portal, central and unclassified lumen holes; edges.py is where build_mesh
    takes classify_walls from, so the classes are the ones the gmsh meshes carry."""
    domain = box(0, 0, 100, 100)
    lumen = np.zeros((100, 100), bool)
    lumen_class = np.zeros((100, 100), np.uint8)
    for x0, y0, x1, y1 in HOLES.values():
        domain = domain.difference(box(x0, y0, x1, y1))
        lumen[y0:y1, x0:x1] = True
    lumen_class[20:40, 20:40] = 1
    lumen_class[20:40, 60:80] = 2
    lumen_class[60:80, 20:40] = 3
    meta = {"um_per_px": 1.0, "origin_um": [0.0, 0.0]}
    chains = edges.edge_chains([domain], edges.frame(meta, lumen.shape), lumen, lumen_class, meta)
    assert on(chains, "portal_wall", HOLES["portal"]) and len(chains["portal_wall"]) == 1
    assert on(chains, "central_wall", HOLES["central"]) and len(chains["central_wall"]) == 1
    assert on(chains, "vessel_wall", HOLES["plain"]) and len(chains["vessel_wall"]) == 1
    assert set(edges.EDGE_TAGS) >= set(chains)
