"""Registration transforms: from a VALIS run into the sidecars, and out to a viewer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from slideviz.data.schema import Registration

SWAP_XY = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def napari_affine(registration: Registration, pixel_size_um: float = 1.0) -> np.ndarray:
    """The transform as napari wants it: row/column order, in the layer's world units."""
    matrix = SWAP_XY @ np.array(registration.matrix, float) @ SWAP_XY
    matrix[:2, 2] *= pixel_size_um
    return matrix


def from_valis_run(run_dir: Path, error_um: float | None = None) -> dict[str, Registration]:
    """Read a VALIS run's transforms.json, keyed by slide stem."""
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
            method="valis-1.2.0 rigid, GradientOD",
            error_um=error_um,
            # how exactly the affine fit the full warp, so a non-affine chain shows up
            residual_px=entry.get("residual_px"),
            # UTC, so a provenance date does not depend on where it was recorded
            registered=datetime.now(UTC).date().isoformat(),
        )
    return found


def write_to_sidecars(
    registrations: dict[str, Registration], slide_dir: Path, write: bool = False
) -> int:
    """Merge registrations into the sidecars of `slide_dir`, matching on file stem."""
    changed = 0
    for name, registration in registrations.items():
        matches = [p for p in slide_dir.glob("*.json") if p.stem == name]
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
