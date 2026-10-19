"""Necrosis on whole slides: the per-patch map (D) snapped to the H&E's edges (E), on tissue without lumina, and the areas.

    uv run python scripts/necrosis_slides.py --slides /data/michelle/slides \
        --maps <predict_slide_patches.py output> --lumen /data/michelle/necrosis_lumen_2026-10-11 \
        --out <dir> --workers 5

E, the readout (her pick, 19 Oct), is D's mean score smoothed with the guided filter on the H&E, at the
settings of the comparison (slideviz.analysis.edge_snap), run in blocks that overlap by twice the window. A
pixel is necrotic where its score reaches the cutoff of the head fitted without its animal. Areas count tissue
the tiles cover: the outline without lumina (the lumen masks of necrosis_lumen.py), levels mapped by their
exact factors. D's area is kept beside E's as a check, and today's tile model's and the drawn polygons'
areas (necrosis_lumen.py's, lumina out) for comparison. Writes one call map per slide and a table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from slideviz.analysis.edge_snap import NECROSIS_EPS, NECROSIS_RADIUS_UM, guided_filter
from slideviz.analysis.prediction import read_lumen
from slideviz.analysis.tissue import (
    mask_from_level,
    on_level,
    pick_level,
    pyramid_factor,
)
from slideviz.io.reader import open_slide

BLOCK = 4096  # level px per side of one filtering block
NOT_MEASURED = 255  # in the call maps: glass, lumen, or tissue no tile covered


def snapped(score: np.ndarray, level: object, origin: tuple[int, int], radius: int) -> np.ndarray:
    """The score map filtered with the H&E as guide, block by block, each block reading twice the window around it."""
    out = np.zeros_like(score)
    pad = 2 * radius
    height, width = score.shape
    for r0 in range(0, height, BLOCK):
        for c0 in range(0, width, BLOCK):
            r1, c1 = min(r0 + BLOCK, height), min(c0 + BLOCK, width)
            er0, ec0, er1, ec1 = max(r0 - pad, 0), max(c0 - pad, 0), min(r1 + pad, height), min(c1 + pad, width)
            guide = np.asarray(level[origin[0] + er0:origin[0] + er1, origin[1] + ec0:origin[1] + ec1])[..., :3] / 255
            filtered = guided_filter(guide, score[er0:er1, ec0:ec1], radius, NECROSIS_EPS)
            out[r0:r1, c0:c1] = filtered[r0 - er0:r1 - er0, c0 - ec0:c1 - ec0]
    return out


def one_slide(job: tuple) -> dict:
    """E and D's calls and areas for one slide; writes its call maps."""
    path, slides, lumen_dir, out = job
    stored = np.load(path)
    meta = json.loads(str(stored["meta"]))
    animal, cutoff, origin = meta["animal"], meta["cutoff"], meta["origin_level_rc"]
    _, levels = open_slide(slides / f"{meta['slide']}.ome.tiff")
    level = levels[meta["level"]]
    width = level.shape[1]
    radius = round(NECROSIS_RADIUS_UM / meta["um_per_px"])

    score = np.nan_to_num(stored["mean"], nan=0.0)
    covered = stored["count"] > 0
    outline = mask_from_level(np.asarray(levels[pick_level(levels)]))
    lumen, _ = read_lumen(lumen_dir / f"{animal}_lumen.npz")
    measured = (covered & on_level(outline, pyramid_factor(width, outline.shape[1]), *origin, score.shape)
                & ~on_level(lumen, pyramid_factor(width, lumen.shape[1]), *origin, score.shape))

    calls = {"E": snapped(score, level, origin, radius) >= cutoff, "D": score >= cutoff}
    n = max(1, int(measured.sum()))
    area = {code: round(float((call & measured).sum() / n), 4) for code, call in calls.items()}
    maps = {f"call_{code}": np.where(measured, call, NOT_MEASURED).astype(np.uint8) for code, call in calls.items()}
    record = {"animal": animal, "slide": meta["slide"], "cutoff": cutoff, "level": meta["level"],
              "um_per_px": meta["um_per_px"], "origin_level_rc": origin, "radius_px": radius,
              "measured_px": int(measured.sum()), "tissue_mm2": round(n * (meta["um_per_px"] / 1000) ** 2, 3),
              "area_E": area["E"], "area_D": area["D"],
              "map": {"written": meta["written"], "code": meta["code"], "file": path.name}}
    np.savez_compressed(out / f"{animal}_necrosis.npz", **maps, meta=json.dumps(record))
    print(f"{animal}: E {area['E']:.1%}, D {area['D']:.1%} of {record['tissue_mm2']} mm2", flush=True)
    return record


def code_version() -> str:
    """The git commit the script ran from, marked dirty if the tree had changes and untracked if it is not in it."""
    here = Path(__file__).parent
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=here)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", Path(__file__).name],
                             capture_output=True, check=False, cwd=here).returncode == 0
    version = found.stdout.strip()
    return "unknown" if not version else version if tracked else f"{version}-untracked"


def main() -> None:
    """Call necrosis on every slide map and write the call maps and the area table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="H&E OME-TIFFs")
    parser.add_argument("--maps", type=Path, required=True, help="predict_slide_patches.py output")
    parser.add_argument("--lumen", type=Path, required=True, help="necrosis_lumen.py output")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty folder")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new --out")
    paths = sorted(args.maps.glob("*_patch_slide.npz"))
    if not paths:
        raise SystemExit(f"no slide maps in {args.maps}")
    args.out.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max(1, min(args.workers, len(paths))), mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(one_slide, [(p, args.slides, args.lumen, args.out) for p in paths]))

    # today's tile model and the drawn polygons, lumina out, from the lumen run
    before = {r["animal"]: r for r in json.loads((args.lumen / "necrosis_lumen.json").read_text())["slides_areas"]}
    for row in rows:
        old = before.get(row["animal"], {})
        row["area_A_tiles"] = old.get("area_predicted_no_lumen")
        row["area_drawn"] = old.get("area_annotated_no_lumen")
    drawn = [r for r in rows if r["area_drawn"]]
    error = {code: round(float(np.mean([abs(r[key] - r["area_drawn"]) for r in drawn])), 4) if drawn else None
             for code, key in (("E", "area_E"), ("D", "area_D"), ("A", "area_A_tiles"))}
    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "readout": "E: D's per-patch mean score, guided filter on the H&E, at the head's cutoff (her pick, 19 Oct)",
        "edge_snap": {"radius_um": NECROSIS_RADIUS_UM, "eps": NECROSIS_EPS, "block_px": BLOCK,
                      "overlap": "twice the radius on every side"},
        "area": "necrotic share of measured tissue: covered by a tile, inside the outline, outside lumina",
        "call_maps": f"<animal>_necrosis.npz: call_E and call_D, uint8 on the map's level from origin_level_rc; "
                     f"1 necrotic, 0 not, {NOT_MEASURED} not measured",
        "mean_abs_area_error_vs_drawn": {"n": len(drawn), **error},
        "error_caveat": "against polygons that stop short of the necrosis edge and include pooled blood; information",
        "slides": rows,
        "inputs": {"maps": str(args.maps), "lumen": str(args.lumen), "slides": str(args.slides)},
    }
    (args.out / "necrosis_slides.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"=== {len(rows)} slides; mean |area - drawn| over {len(drawn)} drawn slides: {error} ===")


if __name__ == "__main__":
    main()
