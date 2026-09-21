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
