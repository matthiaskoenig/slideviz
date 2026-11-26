"""Check the portal and central lumen classes on dosed slides: lumina by class, and those at the necrosis.

    uv run python scripts/wall_class_check.py 000mg_m1 000mg_m2 281mg_m1 --lobules <zonation_layers.py out>

Pericentral necrosis around a central vein that zonation_layers.py classed portal is the sign of a flipped
classification. Per animal, from <stem>_cyp2e1_lobules.npz (lumen_class on the H&E working grid) and the call map
<animal>_necrosis.npz of necrosis_slides.py (call_E, each working pixel by the majority of its calls): the lumina
(connected pieces of lumen_class) by class, portal 1, central 2, other 3; the portal/central ratio; and the lumina
inside or within --radius-um of a necrotic pixel by class. The classification holds on a dosed slide when at most
MAX_NEAR_PORTAL_SHARE of the lumina at the necrosis are portal and its portal/central ratio lies within the range
of the controls (animals of dose 000mg). JSON to stdout, the table to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label, minimum

from slideviz.analysis.prediction import calls_on_grid, read_calls
from slideviz.analysis.zonation import CENTRAL_LUMEN, OTHER_LUMEN, PORTAL_LUMEN
from slideviz.data.provenance import code_version
from slideviz.settings import Settings

RADIUS_UM = 50.0
MAX_NEAR_PORTAL_SHARE = 0.1  # at most 1 in 10 lumina at the necrosis portal
CONTROL_DOSE = "000mg"
CALL_KEY = "call_E"
CLASSES = {"portal": PORTAL_LUMEN, "central": CENTRAL_LUMEN, "other": OTHER_LUMEN}


def by_class(classes: np.ndarray) -> dict[str, int]:
    return {name: int((classes == value).sum()) for name, value in CLASSES.items()}


def count_lumina(lumen_class: np.ndarray, necrotic: np.ndarray, um: float, radius_um: float = RADIUS_UM) -> dict:
    """Lumina by class, the portal/central ratio, and the lumina within radius_um of a necrotic pixel by class.

    A lumen is a connected piece (4-neighbours, as zonation.portality labels them) of lumen_class > 0, of the class
    most of its pixels carry; it is at the necrosis when one of its pixel centres lies within radius_um of a necrotic
    pixel centre (or on one).
    """
    pieces, n = label(lumen_class > 0)
    flat = pieces.ravel()
    votes = np.zeros((n + 1, max(CLASSES.values()) + 1), np.int64)
    np.add.at(votes, (flat[flat > 0], lumen_class.ravel()[flat > 0]), 1)
    classes = votes[1:].argmax(axis=1)
    if n and necrotic.any():
        distance = distance_transform_edt(~necrotic) * um
        nearest = np.asarray(minimum(distance, pieces, np.arange(1, n + 1)), float).reshape(n)
        near = nearest <= radius_um
    else:
        near = np.zeros(n, bool)
    counts, at_necrosis = by_class(classes), by_class(classes[near])
    total_near = sum(at_necrosis.values())
    tagged_near = at_necrosis["portal"] + at_necrosis["central"]
    return {
        "lumina": counts,
        "portal_central_ratio": counts["portal"] / counts["central"] if counts["central"] else None,
        "near_necrosis": at_necrosis,
        "near_portal_share": at_necrosis["portal"] / total_near if total_near else None,
        "near_portal_of_tagged": at_necrosis["portal"] / tagged_near if tagged_near else None,
    }


def gate(rows: list[dict]) -> dict | None:
    """The verdict per dosed slide against the controls' range of portal/central ratios; None without controls."""
    controls = [row for row in rows if row["animal"].startswith(CONTROL_DOSE)]
    ratios = [row["portal_central_ratio"] for row in controls if row["portal_central_ratio"] is not None]
    if not ratios:
        return None
    low, high = min(ratios), max(ratios)
    dosed = []
    for row in rows:
        if row in controls:
            continue
        ratio, share = row["portal_central_ratio"], row["near_portal_share"]
        in_range = ratio is not None and low <= ratio <= high
        # no lumen at the necrosis: no sign of a flip
        share_ok = share is None or share <= MAX_NEAR_PORTAL_SHARE
        dosed.append({"animal": row["animal"], "ratio_in_range": in_range, "near_portal_ok": share_ok,
                      "holds": in_range and share_ok})
    return {"controls": [row["animal"] for row in controls], "control_ratio_range": [low, high],
            "max_near_portal_share": MAX_NEAR_PORTAL_SHARE, "dosed": dosed,
            "holds": all(row["holds"] for row in dosed)}


def one_file(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern} in {directory}, found {[p.name for p in found]}")
    return found[0]


def check(animal: str, lobules_dir: Path, predictions: Path, radius_um: float) -> dict:
    """The counts of one animal."""
    lobules = one_file(lobules_dir, f"*_{animal}_cyp2e1_lobules.npz")
    necrosis = predictions / f"{animal}_necrosis.npz"
    if not necrosis.exists():
        raise SystemExit(f"the necrosis call map is missing: {necrosis}")
    with np.load(lobules) as stored:
        if "lumen_class" not in stored:
            raise SystemExit(f"{lobules} has no lumen_class; rebuild it with scripts/zonation_layers.py")
        classes, reference = stored["lumen_class"], str(stored["reference"])
        um, origin = float(stored["working_um_per_px"]), tuple(stored["working_origin_um"])
    calls, meta = read_calls(necrosis, CALL_KEY)
    if meta["slide"] != reference:
        raise SystemExit(f"{necrosis.name} belongs to {meta['slide']}, {lobules.name} to {reference}")
    necrotic, measured = calls_on_grid(calls, meta, um, origin, slice(0, classes.shape[0]), slice(0, classes.shape[1]))
    del calls
    return {"animal": animal, "slide": reference, "lobules": str(lobules), "necrosis": str(necrosis),
            "necrosis_cutoff": meta.get("cutoff"), "um_per_px": um,
            "necrotic_share": float(necrotic.sum() / max(measured.sum(), 1)),
            **count_lumina(classes, necrotic, um, radius_um)}


def number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def table(rows: list[dict], verdict: dict | None) -> str:
    """The rows as a Markdown table, the gate under it."""
    header = ("| animal | necrotic % | portal | central | other | portal/central | near portal | near central | "
              "near other | near portal share | near portal / (portal + central) |")
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        n, a = row["near_necrosis"], row["lumina"]
        lines.append(f"| {row['animal']} | {100 * row['necrotic_share']:.1f} | {a['portal']} | {a['central']} | "
                     f"{a['other']} | {number(row['portal_central_ratio'])} | {n['portal']} | {n['central']} | "
                     f"{n['other']} | {number(row['near_portal_share'], 3)} | "
                     f"{number(row['near_portal_of_tagged'], 3)} |")
    if verdict:
        low, high = verdict["control_ratio_range"]
        lines.append(f"\ncontrols {', '.join(verdict['controls'])}: portal/central {low:.2f} to {high:.2f}")
        for row in verdict["dosed"]:
            lines.append(f"{row['animal']}: ratio in range {row['ratio_in_range']}, near portal share at most "
                         f"{verdict['max_near_portal_share']} {row['near_portal_ok']}: "
                         f"{'holds' if row['holds'] else 'does not hold'}")
    return "\n".join(lines)


def main() -> None:
    """Count the lumina of each animal and print the JSON and the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("animals", nargs="+", help="animal names of the prediction files, e.g. 281mg_m1")
    parser.add_argument("--lobules", type=Path, required=True,
                        help="folder of zonation_layers.py <stem>_lobules.npz files (with lumen_class)")
    parser.add_argument("--predictions", type=Path, help="folder of the <animal>_necrosis.npz call maps "
                        "(default SLIDEVIZ_DATA/predictions)")
    parser.add_argument("--radius-um", type=float, default=RADIUS_UM, help="lumina this close count as at the necrosis")
    args = parser.parse_args()
    predictions = args.predictions or Settings().predictions_dir()
    if predictions is None:
        raise SystemExit("give --predictions or set SLIDEVIZ_DATA")
    rows = [check(animal, args.lobules, predictions, args.radius_um) for animal in args.animals]
    verdict = gate(rows)
    print(json.dumps({"radius_um": args.radius_um, "call": CALL_KEY, "code": code_version(Path(__file__)),
                      "slides": rows, "gate": verdict}, indent=2))
    print(table(rows, verdict), file=sys.stderr)


if __name__ == "__main__":
    main()
