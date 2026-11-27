"""Tag names and values and the size field constants shared by the mesh scripts; no dependencies (no shapely, gmsh)."""

AREA_TAGS = {"surviving": 1, "necrotic": 2, "tissue": 3}
EDGE_TAGS = {"vessel_wall": 11, "section_edge": 12, "field_cut": 13, "necrosis_border": 14, "portal_wall": 15,
             "central_wall": 16}
REFINED = ("vessel_wall", "portal_wall", "central_wall", "section_edge", "necrosis_border")  # the square's cut sides are artificial
DIST_MIN, DIST_MAX = 2, 10  # small size up to DIST_MIN x small from a refined edge, large size from DIST_MAX x large
