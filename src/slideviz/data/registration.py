"""Registration transforms: from a VALIS run into the sidecars, and out to a viewer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from slideviz.data.schema import NonRigid, Registration

FIELD_DIR = "transforms"  # beside the sidecars; NonRigid.path points into it

SWAP_XY = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def napari_affine(registration: Registration, pixel_size_um: float = 1.0) -> np.ndarray:
    """The transform as napari wants it: row/column order, in the layer's world units."""
    matrix = SWAP_XY @ np.array(registration.matrix, float) @ SWAP_XY
    matrix[:2, 2] *= pixel_size_um
    return matrix


VALIS_METHOD = "valis-1.2.0 rigid, GradientOD"


def from_valis_run(
    run_dir: Path, error_um: float | None = None, method: str = VALIS_METHOD
) -> dict[str, Registration]:
    """Read a run's transforms.json, keyed by slide stem.

    `method` records what produced it, since outline_register.py writes this file too.
    """
    data = json.loads((run_dir / "transforms.json").read_text())
    reference = Path(data["reference"]).name.split(".")[0]

    found = {}
    for name, entry in data["slides"].items():
        if name == reference:
            continue
        found[name] = Registration(
            reference=reference,
            matrix=entry["matrix"],
            slide_shape_rc=entry["slide_shape_rc"],
            method=method,
            error_um=error_um,
            # how exactly the affine fit the full warp
            residual_px=entry.get("residual_px"),
            outline_dice=entry.get("outline_dice"),
            registered=datetime.now(UTC).date().isoformat(),  # UTC, for provenance
        )
    return found


def write_to_sidecars(
    registrations: dict[str, Registration], slide_dir: Path, write: bool = False
) -> int:
    """Merge registrations into the sidecars of `slide_dir`, matching on file stem."""
    changed = 0
    for name, registration in registrations.items():
        # <name>.ome.json keeps the .ome in its stem, so cut at the first dot
        matches = [p for p in slide_dir.glob("*.json") if p.name.split(".")[0] == name]
        if not matches:
            print(f"  ! no sidecar for {name}")
            continue

        sidecar = matches[0]
        data = json.loads(sidecar.read_text())
        data["registration"] = registration.model_dump(exclude_none=True)
        changed += 1
        print(f"  {name}: registered to {registration.reference}")
        if write:
            sidecar.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def sha256(path: Path) -> str:
    """Hex digest of a file, read in chunks."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_nonrigid(summary_path: Path, slide_dir: Path, write: bool = False) -> NonRigid:
    """Point a sidecar at its non-rigid field, refusing one computed on another matrix."""
    import shutil

    summary = json.loads(summary_path.read_text())
    field = summary_path.parent / summary["field"]
    matches = [p for p in slide_dir.glob("*.json") if p.name.split(".")[0] == summary["slide"]]
    if not matches:
        raise FileNotFoundError(f"no sidecar for {summary['slide']} in {slide_dir}")
    sidecar = matches[0]

    data = json.loads(sidecar.read_text())
    registration = Registration(**data["registration"])
    with np.load(field) as stored:
        if not np.allclose(stored["matrix"], registration.matrix):
            raise ValueError(f"{summary['slide']}: the field was computed on another matrix "
                             "than the sidecar carries, so it no longer applies")

    nonrigid = NonRigid(
        path=f"{FIELD_DIR}/{field.name}",
        sha256=sha256(field),
        grid_um_per_px=summary["grid_um_per_px"],
        grid_shape_rc=summary["grid_shape_rc"],
        method=summary["method"],
        median_um=summary["median_um"],
        p95_um=summary["p95_um"],
        registered=summary["registered"],
    )
    print(f"  {summary['slide']}: {nonrigid.path}, median {nonrigid.median_um} µm")
    if write:
        (slide_dir / FIELD_DIR).mkdir(exist_ok=True)
        shutil.copy2(field, slide_dir / nonrigid.path)
        data["registration"]["nonrigid"] = nonrigid.model_dump(exclude_none=True)
        sidecar.write_text(json.dumps(data, indent=2) + "\n")
    return nonrigid


def main() -> None:
    """Attach every non-rigid field in a directory to the sidecars of another."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("fields", type=Path, help="directory of *_nonrigid.json summaries")
    parser.add_argument("slide_dir", type=Path, help="directory of the slides' sidecars")
    parser.add_argument("--write", action="store_true", help="without it, only report")
    args = parser.parse_args()

    summaries = sorted(args.fields.glob("*_nonrigid.json"))
    for summary in summaries:
        attach_nonrigid(summary, args.slide_dir, write=args.write)
    print(f"{len(summaries)} fields {'attached' if args.write else 'checked, nothing written'}")


if __name__ == "__main__":
    main()
