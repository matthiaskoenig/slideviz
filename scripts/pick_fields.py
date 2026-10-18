"""One 1 mm field per H&E slide for comparing necrosis maps: the manuscript figure's zoom where it has one, else by a rule.

    uv run --no-sync --with matplotlib python scripts/pick_fields.py --slides /data/michelle/slides \
        --predictions /data/michelle/slides/predictions --out <fields.json>

The figure's zoom centres come from manuscript_figure.py, which needs matplotlib.

The rule works on today's 75 um tile calls (the nested predictions): the 1 mm square, nearly all tissue,
holding the most necrosis edge (a called tile beside a tissue tile not called), and of those the most
tissue. A slide without any edge
takes the square nearest its tissue's centre. Centres are written in H&E level-0 px (row, col), as
export_field_tiles.py --field takes them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from manuscript_figure import BLOCKS, ZOOM_UM
from scipy.ndimage import convolve

from slideviz.analysis.tiling import TILE_UM
from slideviz.io.reader import open_slide

FULL = 0.8  # share of the square's tile positions that must hold tissue


def tile_grid(prediction: dict) -> tuple[np.ndarray, np.ndarray]:
    """Which tile positions hold tissue, and which of those today's model calls necrotic."""
    shape = (max(t["row"] for t in prediction["tiles"]) + 1, max(t["col"] for t in prediction["tiles"]) + 1)
    present, called = np.zeros(shape, bool), np.zeros(shape, bool)
    for t in prediction["tiles"]:
        present[t["row"], t["col"]] = True
        called[t["row"], t["col"]] = t["predicted"] >= prediction["cutoff"]
    return present, called


def necrosis_edge(present: np.ndarray, called: np.ndarray) -> np.ndarray:
    """Called tiles with a tissue tile beside them, above, below, left or right, that is not called."""
    open_tissue = np.pad(present & ~called, 1)
    beside = open_tissue[:-2, 1:-1] | open_tissue[2:, 1:-1] | open_tissue[1:-1, :-2] | open_tissue[1:-1, 2:]
    return called & beside


def pick(prediction: dict) -> tuple[tuple[int, int], dict]:
    """The centre tile (row, col) of the rule's square, and what decided it."""
    present, called = tile_grid(prediction)
    edge = necrosis_edge(present, called)
    side = round(ZOOM_UM / TILE_UM) | 1  # odd, so the square has a centre tile
    window = np.ones((side, side))
    tissue = convolve(present.astype(float), window, mode="constant")
    edges = convolve(edge.astype(float), window, mode="constant")
    full = tissue >= FULL * side * side
    if not full.any():
        raise ValueError(f"{prediction['animal']}: no {ZOOM_UM:.0f} um square is {FULL:.0%} tissue")

    if edges[full].max() > 0:
        # the most edge, and of those squares the most tissue
        score = np.where(full, edges * (side * side + 1) + tissue, -1)
        centre = np.unravel_index(np.argmax(score), edges.shape)
        reason = "most necrosis edge"
    else:
        rows, cols = np.nonzero(present)
        distance = np.hypot(*np.meshgrid(np.arange(present.shape[0]) - rows.mean(),
                                         np.arange(present.shape[1]) - cols.mean(), indexing="ij"))
        centre = np.unravel_index(np.argmin(np.where(full, distance, np.inf)), distance.shape)
        reason = "no necrosis edge, nearest the tissue centre"
    r, c = (int(v) for v in centre)
    return (r, c), {"rule": reason, "edge_tiles": int(edges[r, c]), "tissue_tiles": int(tissue[r, c]),
                    "square_tiles": side * side}


def code_version() -> str:
    """The git commit the script ran from, marked dirty if the tree had changes."""
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=Path(__file__).parent)
    return found.stdout.strip() or "unknown"


def main() -> None:
    """Choose every slide's field and write the centres with what chose them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="H&E OME-TIFFs")
    parser.add_argument("--predictions", type=Path, required=True, help="nested tile predictions, one per slide")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    figure = dict(BLOCKS.values())
    fields = {}
    for path in sorted(args.predictions.glob("*_predictions.json")):
        prediction = json.loads(path.read_text())
        animal = prediction["animal"]
        (r, c), why = pick(prediction)
        if animal in figure:
            centre = figure[animal]
            why = {"rule": "manuscript figure zoom", "rule_would_pick_tile": [r, c]}
        else:
            # the tile centre on its level, then level 0 by the pyramid's own scale, as export_field_tiles.py reads it
            _, levels = open_slide(args.slides / f"{prediction['slide']}.ome.tiff")
            to_level0 = levels[0].shape[1] / levels[prediction["level"]].shape[1]
            half = prediction["size_px"] / 2
            centre = (round((r * prediction["size_px"] + half) * to_level0),
                      round((c * prediction["size_px"] + half) * to_level0))
        fields[animal] = {"slide": prediction["slide"], "centre_level0_rc": [int(v) for v in centre], **why,
                          "prediction": path.name, "prediction_written": prediction["written"],
                          "cutoff": prediction["cutoff"]}
        print(f"{animal}: {centre[0]:>6} {centre[1]:>6}  {why['rule']}"
              + (f", {why['edge_tiles']} edge tiles" if "edge_tiles" in why else ""))

    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "size_um": ZOOM_UM,
        "rule": (f"the manuscript figure's zoom where it has one; else the {ZOOM_UM:.0f} um square of 75 um tile "
                 f"positions, at least {FULL:.0%} tissue, holding the most tiles that today's model calls necrotic "
                 "beside a tissue tile it does not call, and of those the most tissue; without any, the square "
                 "nearest the tissue's centre"),
        "predictions": str(args.predictions),
        "fields": fields,
        "export_args": " ".join(f"--field {a} {f['centre_level0_rc'][0]} {f['centre_level0_rc'][1]}"
                                for a, f in fields.items()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(fields)} fields to {args.out}")


if __name__ == "__main__":
    main()
