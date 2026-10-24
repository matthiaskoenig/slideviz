"""Cut vessel lumina out of the necrosis predictions: a lumen mask per H&E slide and areas without lumina.

Example:
    uv run python scripts/necrosis_lumen.py --slides /data/michelle/slides \
        --predictions /data/michelle/slides/predictions --out /data/michelle/necrosis_lumen_run --workers 10
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from skimage.transform import resize

from slideviz.analysis import dab, tissue
from slideviz.analysis.dab import glass_colour, glass_like, stain_od
from slideviz.analysis.tiling import pick_level_um
from slideviz.analysis.tissue import lumen_mask, mask_from_level, pick_level
from slideviz.data.provenance import code_version
from slideviz.data.registration import sha256
from slideviz.io.reader import open_slide

LUMEN_UM = dab.WORK_UM


def level_um(levels: list, level0_um: float, index: int) -> float:
    """Pixel size of one pyramid level."""
    return level0_um * levels[0].shape[1] / levels[index].shape[1]


def slide_lumen(path: Path) -> tuple[np.ndarray, float, list[float]]:
    """Lumina inside the tissue outline of one H&E slide at the level nearest LUMEN_UM, and every level's um/px."""
    info, levels = open_slide(path)
    coarse = np.asarray(levels[pick_level(levels)])[..., :3]
    outline = mask_from_level(coarse)
    sizes = [level_um(levels, info.pixel_size_um, i) for i in range(len(levels))]
    index = pick_level_um([level.shape for level in levels], info.pixel_size_um, LUMEN_UM)
    rgb = np.asarray(levels[index])[..., :3]
    inside = resize(outline, rgb.shape[:2], order=0).astype(bool)
    lumen = lumen_mask(glass_like(stain_od(rgb, glass_colour(coarse, outline))), sizes[index]) & inside
    return lumen, sizes[index], sizes


def tile_shares(record: dict, lumen: np.ndarray, lumen_um: float, level_size_um: float) -> np.ndarray:
    """Share of each prediction tile covered by lumen, in the prediction file's tile order."""
    scale = level_size_um / lumen_um
    size = record["size_px"] * scale
    return np.array([lumen[int(t["y"] * scale):int(t["y"] * scale + size),
                           int(t["x"] * scale):int(t["x"] * scale + size)].mean()
                     for t in record["tiles"]], dtype=np.float32)


def areas(record: dict, shares: np.ndarray) -> dict:
    """Predicted and labelled necrotic area, as tile shares and as non-lumen tissue."""
    tiles = record["tiles"]
    predicted = np.array([t["predicted"] >= record["cutoff"] for t in tiles])
    labelled = np.array([t["necrosis"] >= record["positive_threshold"] for t in tiles])
    keep = 1.0 - shares.astype(float)
    return {
        "area_predicted": round(float(predicted.mean()), 4),
        "area_annotated": round(float(labelled.mean()), 4),
        "area_predicted_no_lumen": round(float(keep[predicted].sum() / keep.sum()), 4),
        "area_annotated_no_lumen": round(float(keep[labelled].sum() / keep.sum()), 4),
        "lumen_of_tissue": round(float(shares.mean()), 4),
    }


def one_slide(job: tuple) -> dict:
    """Write one slide's lumen file; the row for the run record."""
    path, slides, out, code = job
    record = json.loads(path.read_text())
    lumen, lumen_um, sizes = slide_lumen(slides / f"{record['slide']}.ome.tiff")
    shares = tile_shares(record, lumen, lumen_um, sizes[record["level"]])
    row = areas(record, shares)
    for name in ("area_predicted", "area_annotated"):
        if row[name] != record[name]:
            raise RuntimeError(f"{path.name}: {name} {row[name]} does not reproduce the file's {record[name]}")
    meta = {
        "slide": record["slide"],
        "animal": record["animal"],
        "prediction": path.name,
        "prediction_sha256": sha256(path),
        "rule": "tissue.lumen_mask on dab.glass_like inside the tissue outline",
        "settings": {"LUMEN_UM": LUMEN_UM, "GLASS_OD": dab.GLASS_OD,
                     "LUMEN_OPEN_UM": tissue.LUMEN_OPEN_UM, "LUMEN_MIN_UM2": tissue.LUMEN_MIN_UM2},
        "code": code,
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        **row,
    }
    np.savez_compressed(out / f"{record['animal']}_lumen.npz", lumen=lumen, lumen_um=lumen_um,
                        tile_lumen=shares, meta=json.dumps(meta))
    print(f"{record['animal']}: predicted {row['area_predicted']:.1%} -> {row['area_predicted_no_lumen']:.1%}",
          flush=True)
    return {"animal": record["animal"], **row}


def main() -> None:
    """Write a lumen file per prediction and a run record with the areas."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="H&E OME-TIFFs")
    parser.add_argument("--predictions", type=Path, required=True, help="nested necrosis predictions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.predictions.glob("*_predictions.json"))
    if not paths:
        raise SystemExit(f"no prediction files in {args.predictions}")
    code = code_version(Path(__file__))
    with Pool(max(1, min(args.workers, len(paths)))) as pool:
        rows = pool.map(one_slide, [(p, args.slides, args.out, code) for p in paths])

    necrotic = [r for r in rows if r["area_annotated"] > 0]
    record = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code,
        "slides": str(args.slides),
        "predictions": str(args.predictions),
        "area": "non-lumen tissue in necrotic tiles over all non-lumen tissue",
        "mean_abs_error": {
            "slides": len(necrotic),
            "with_lumen": round(float(np.mean([abs(r["area_predicted"] - r["area_annotated"])
                                               for r in necrotic])), 4),
            "no_lumen": round(float(np.mean([abs(r["area_predicted_no_lumen"] - r["area_annotated_no_lumen"])
                                             for r in necrotic])), 4),
        },
        "slides_areas": rows,
    }
    (args.out / "necrosis_lumen.json").write_text(json.dumps(record, indent=1))
    print(f"{len(rows)} slides, mean area error {record['mean_abs_error']}, written to {args.out}")


if __name__ == "__main__":
    main()
