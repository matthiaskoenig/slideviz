"""Hand landmarks for registration error: place matching points, then score the transforms.

    uv run python scripts/landmarks.py place 281mg_m2 --slides <APAP_tiff> --out <landmarks>
    uv run python scripts/landmarks.py score --slides <APAP_tiff> --out <landmarks> \
        --fields density=<nonrigid dir> haematoxylin=<nonrigid dir>

Points are stored as level-0 (row, col) pixels in each slide's own frame; pair i is the
i-th H&E point with the i-th CYP2E1 point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from slideviz.data.registration import napari_affine
from slideviz.data.schema import Registration
from slideviz.io.reader import open_slide

VIEW_UM_PER_PX = 1.8
CACHE = Path.home() / ".cache" / "slideviz" / "landmarks"
POINT_UM = 40


def sidecar(slides: Path, slide: str) -> dict:
    """A slide's sidecar, named <stem>.json or <stem>.ome.json."""
    for name in (f"{slide}.json", f"{slide}.ome.json"):
        if (slides / name).exists():
            return json.loads((slides / name).read_text())
    raise FileNotFoundError(f"no sidecar for {slide} in {slides}")


def view(slides: Path, slide: str) -> tuple[np.ndarray, float, float]:
    """The level nearest VIEW_UM_PER_PX in memory, cached: (image, level-0 um/px, factor)."""
    info, levels = open_slide(slides / f"{slide}.ome.tiff")
    factors = [levels[0].shape[1] / level.shape[1] for level in levels]
    index = min(range(len(levels)),
                key=lambda i: abs(info.pixel_size_um * factors[i] - VIEW_UM_PER_PX))
    path = CACHE / f"{slide}_level{index}.npy"
    if path.exists():
        image = np.load(path)
    else:
        image = np.asarray(levels[index])
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, image)
    return image, info.pixel_size_um, factors[index]


def place(block: str, slides: Path, out: Path) -> None:
    """Open H&E and rigidly placed CYP2E1 side by side with a points layer on each."""
    import napari

    he, cyp = f"mouse_apap_{block}_he", f"mouse_apap_{block}_cyp2e1"
    he_image, he_um, he_factor = view(slides, he)
    cyp_image, cyp_um, cyp_factor = view(slides, cyp)
    registration = Registration(**sidecar(slides, cyp)["registration"])
    affine = napari_affine(registration, he_um)

    path = out / f"{block}.json"
    stored = json.loads(path.read_text()) if path.exists() else {"he": [], "cyp2e1": []}

    viewer = napari.Viewer(title=f"{block} landmarks: H&E left, CYP2E1 right")
    viewer.add_image(he_image, name="H&E", rgb=True, scale=(he_um * he_factor,) * 2)
    he_points = viewer.add_points(np.array(stored["he"]).reshape(-1, 2), name="H&E points",
                                  scale=(he_um,) * 2, size=POINT_UM / he_um,
                                  face_color="lime", border_color="black")
    viewer.add_image(cyp_image, name="CYP2E1", rgb=True, scale=(cyp_um * cyp_factor,) * 2,
                     affine=affine)
    cyp_points = viewer.add_points(np.array(stored["cyp2e1"]).reshape(-1, 2),
                                   name="CYP2E1 points", scale=(cyp_um,) * 2, affine=affine,
                                   size=POINT_UM / cyp_um, face_color="magenta",
                                   border_color="black")
    for layer in (he_points, cyp_points):
        layer.feature_defaults = {}
        layer.mode = "add"
    viewer.grid.enabled = True
    viewer.grid.stride = 2  # each image with its points in one panel

    def save(_=None) -> None:
        """Write both point lists and show the pair count."""
        record = {"block": block, "he_slide": he, "cyp2e1_slide": cyp,
                  "he": he_points.data.round(1).tolist(),
                  "cyp2e1": cyp_points.data.round(1).tolist()}
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
        viewer.status = f"{len(he_points.data)} H&E, {len(cyp_points.data)} CYP2E1 points saved"

    he_points.events.data.connect(save)
    cyp_points.events.data.connect(save)
    viewer.layers.selection.active = he_points
    napari.run()


def field_at(field: dict, points_xy: np.ndarray, reference_um: float) -> np.ndarray:
    """The displacement in reference level-0 px at reference points (x, y)."""
    from scipy.ndimage import map_coordinates

    grid = [points_xy[:, 1] * reference_um / field["grid_um_per_px"],
            points_xy[:, 0] * reference_um / field["grid_um_per_px"]]
    dx = map_coordinates(field["dx_um"], grid, order=1, mode="nearest")
    dy = map_coordinates(field["dy_um"], grid, order=1, mode="nearest")
    return np.stack([dx, dy], axis=1) / reference_um


def score(slides: Path, out: Path, fields: dict[str, Path]) -> None:
    """Landmark error per block for the rigid matrix and each set of fields, in um."""
    rows = []
    for path in sorted(out.glob("*.json")):
        record = json.loads(path.read_text())
        pairs = min(len(record["he"]), len(record["cyp2e1"]))
        if pairs == 0:
            continue
        he_um = view_um(slides, record["he_slide"])
        cyp_um = view_um(slides, record["cyp2e1_slide"])
        p = np.array(record["he"])[:pairs, ::-1]  # (x, y) reference px
        q = np.array(record["cyp2e1"])[:pairs, ::-1]  # (x, y) moving px

        errors = {}
        for name, directory in {"rigid": None, **fields}.items():
            if directory is None:
                matrix = np.array(Registration(
                    **sidecar(slides, record["cyp2e1_slide"])["registration"]).matrix)
                shifted = p
            else:
                with np.load(directory / f"{record['cyp2e1_slide']}_nonrigid.npz") as stored:
                    field = {k: stored[k] for k in ("dx_um", "dy_um", "matrix")}
                    field["grid_um_per_px"] = float(stored["grid_um_per_px"])
                matrix = field["matrix"]
                shifted = p + field_at(field, p, he_um)
            inverse = np.linalg.inv(np.asarray(matrix, float))
            predicted = shifted @ inverse[:2, :2].T + inverse[:2, 2]
            errors[name] = np.hypot(*(predicted - q).T) * cyp_um
        rows.append((record["block"], pairs, errors))

    names = ["rigid", *fields]
    print(f"{'block':<10}{'pairs':>6}" + "".join(f"{n:>14}" for n in names) + "   (median um)")
    for block, pairs, errors in rows:
        print(f"{block:<10}{pairs:>6}" + "".join(f"{np.median(errors[n]):>14.1f}" for n in names))
    for n in names:
        pooled = np.concatenate([e[n] for _, _, e in rows]) if rows else np.array([np.nan])
        print(f"{'all':<10}{len(pooled):>6}{n:>14}  median {np.median(pooled):.1f}, "
              f"p90 {np.percentile(pooled, 90):.1f} um")
    (out / "scores.json").write_text(json.dumps(
        {block: {n: [round(float(v), 1) for v in e[n]] for n in names}
         for block, _, e in rows}, indent=1) + "\n")


def view_um(slides: Path, slide: str) -> float:
    """A slide's level-0 pixel size."""
    info, _ = open_slide(slides / f"{slide}.ome.tiff")
    return info.pixel_size_um


def main() -> None:
    """Place landmarks for one block, or score all placed blocks."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("place", "score", "cache"))
    parser.add_argument("block", nargs="*", help="e.g. 281mg_m2")
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="landmark directory")
    parser.add_argument("--fields", nargs="*", default=[],
                        help="name=directory of <slide>_nonrigid.npz")
    args = parser.parse_args()

    if args.command == "place":
        place(args.block[0], args.slides, args.out)
    elif args.command == "cache":
        for block in args.block:
            for stain in ("he", "cyp2e1"):
                view(args.slides, f"mouse_apap_{block}_{stain}")
                print(f"cached {block} {stain}", flush=True)
    else:
        fields = dict(item.split("=", 1) for item in args.fields)
        score(args.slides, args.out, {k: Path(v) for k, v in fields.items()})


if __name__ == "__main__":
    main()
