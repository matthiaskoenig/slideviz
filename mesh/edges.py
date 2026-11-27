"""Tagged boundary chains of areas: the edge features of a polygons/1 file and their wall classes.

The areas' rings are the only source of coordinates. Each ring segment is classified (between two areas, on the
square's sides, around a vessel, the section's edge) and the segments of one tag are joined to chains, so a chain
repeats the very same vertices as the rings it lies on, which slidesim's mesher requires (no snapping).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import shapely
from shapely.geometry import MultiLineString
from tags import EDGE_TAGS

WALL_CLASSES = {1: "portal_wall", 2: "central_wall"}  # lumen_class values that name a wall; 3 and 0 stay vessel_wall
CLASS_REACH_UM = 10.0  # lumen class pixels this close to a wall's edge midpoints vote for its class
EPS_UM = 1e-6


def frame(meta: dict, shape_rc: tuple[int, int]) -> tuple[float, float, float, float]:
    """The square's outer pixel edges (x0, y0, x1, y1) in um."""
    um, (oy, ox) = meta["um_per_px"], meta["origin_um"]
    return ox - um / 2, oy - um / 2, ox + (shape_rc[1] - 0.5) * um, oy + (shape_rc[0] - 0.5) * um


def on_side(a: tuple, b: tuple, box: tuple) -> bool:
    """An edge piece lying on one of the square's sides."""
    x0, y0, x1, y1 = box
    return any(abs(a[i] - v) < EPS_UM and abs(b[i] - v) < EPS_UM for i, v in ((0, x0), (0, x1), (1, y0), (1, y1)))


def lumen_beside(a: tuple, b: tuple, lumen: np.ndarray, meta: dict) -> bool:
    """The lumen mask one pixel to either side of an edge piece's midpoint."""
    um, (oy, ox) = meta["um_per_px"], meta["origin_um"]
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = np.hypot(dx, dy) or 1.0
    for side in (1, -1):
        r = round((my + side * dx / length * um - oy) / um)
        c = round((mx - side * dy / length * um - ox) / um)
        if 0 <= r < lumen.shape[0] and 0 <= c < lumen.shape[1] and lumen[r, c]:
            return True
    return False


def wall_class(chain: list[tuple], lumen_class: np.ndarray, meta: dict) -> str:
    """The wall name of a vessel-wall chain, by majority of the portal and central lumen pixels near its edge midpoints.

    Pixels are counted once, however many midpoints reach them. A tie or no classified pixel leaves vessel_wall.
    """
    um, (oy, ox) = meta["um_per_px"], meta["origin_um"]
    reach = int(np.ceil(CLASS_REACH_UM / um))
    near = set()
    for a, b in chain:
        r, c = round(((a[1] + b[1]) / 2 - oy) / um), round(((a[0] + b[0]) / 2 - ox) / um)
        for i in range(max(r - reach, 0), min(r + reach + 1, lumen_class.shape[0])):
            for j in range(max(c - reach, 0), min(c + reach + 1, lumen_class.shape[1])):
                if np.hypot(i - r, j - c) * um <= CLASS_REACH_UM:
                    near.add((i, j))
    votes = {name: 0 for name in WALL_CLASSES.values()}
    for i, j in near:
        if lumen_class[i, j] in WALL_CLASSES:
            votes[WALL_CLASSES[lumen_class[i, j]]] += 1
    (first, n1), (_, n2) = sorted(votes.items(), key=lambda kv: -kv[1])
    return first if n1 > n2 else "vessel_wall"


def classify_walls(walls: list[tuple], xy: dict, lumen_class: np.ndarray, meta: dict) -> dict:
    """Vessel-wall edge pieces (tag, corner a, corner b) by wall name, each connected chain of them named as one."""
    parent = {}

    def find(k):
        """Chain representative of corner k."""
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for _, a, b in walls:
        parent[find(a)] = find(b)
    chains: dict = {}
    for item in walls:
        chains.setdefault(find(item[1]), []).append(item)
    named = {name: [] for name in ("vessel_wall", *WALL_CLASSES.values())}
    for items in chains.values():
        name = wall_class([(xy[a], xy[b]) for _, a, b in items], lumen_class, meta)
        named[name].extend(tag for tag, _, _ in items)
    return named


def ring_segments(polygon) -> list[tuple[tuple, tuple, bool]]:
    """The segments (a, b, is_hole) of a polygon's rings, a and b as (x, y) tuples exactly as stored."""
    out = []
    for ring, hole in [(polygon.exterior, False), *[(r, True) for r in polygon.interiors]]:
        coords = [(float(x), float(y)) for x, y in ring.coords]
        out += [(a, b, hole) for a, b in pairwise(coords) if a != b]
    return out


def edge_chains(areas: list, box: tuple, lumen: np.ndarray, lumen_class: np.ndarray, meta: dict) -> dict[str, list]:
    """Every boundary segment of the areas in one edge tag; its chains (lists of (x, y)) by tag name.

    A segment of two areas is necrosis_border, one on the square's sides field_cut, one around a hole or beside a
    lumen a wall (portal_wall or central_wall by the class vote of its connected walls, else vessel_wall), else
    section_edge. Segments match by their exact coordinates, so the areas must share their borders vertex for vertex.
    """
    seen: dict[tuple, list] = {}
    for polygon in areas:
        for a, b, hole in ring_segments(polygon):
            seen.setdefault((min(a, b), max(a, b)), []).append(hole)
    for (a, b), holes in seen.items():
        if len(holes) > 2:
            raise ValueError(f"the segment {a} - {b} lies on {len(holes)} rings; at most two areas may share a border "
                             "(overlapping areas or a doubled ring)")
    groups: dict[str, list] = {name: [] for name in EDGE_TAGS}
    walls = []
    for (a, b), holes in seen.items():
        if len(holes) == 2:
            groups["necrosis_border"].append((a, b))
        elif on_side(a, b, box):
            groups["field_cut"].append((a, b))
        elif holes[0] or lumen_beside(a, b, lumen, meta):
            walls.append(((a, b), a, b))
        else:
            groups["section_edge"].append((a, b))
    xy = {k: k for _, a, b in walls for k in (a, b)}
    for name, segments in classify_walls(walls, xy, lumen_class, meta).items():
        groups[name].extend(segments)
    chains = {}
    for name, segments in groups.items():
        if segments:
            merged = shapely.line_merge(MultiLineString(segments))
            chains[name] = [[(float(x), float(y)) for x, y in line.coords] for line in shapely.get_parts(merged)]
    return chains
