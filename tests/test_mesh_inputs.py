"""The lumen classes of a zonation map resampled onto a mesh_inputs window, through the H&E um frame."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("slideviz")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mesh_inputs

WORKING = np.arange(36, dtype=np.uint8).reshape(6, 6) % 4  # distinct-enough classes 0-3 on a 6 x 6 grid of 2 um


def lobules(tmp_path: Path, reference: str = "he", origin: tuple = (0.0, 0.0)) -> Path:
    path = tmp_path / "x_lobules.npz"
    np.savez_compressed(path, lumen_class=WORKING, working_um_per_px=2.0, working_origin_um=np.array(origin),
                        reference=reference)
    return path


def test_identity(tmp_path: Path) -> None:
    out = mesh_inputs.lumen_classes(lobules(tmp_path), "he", 0, slice(0, 6), slice(0, 6), 2.0)
    np.testing.assert_array_equal(out, WORKING)


def test_coarser_target(tmp_path: Path) -> None:
    # 4 um pixels: target pixel (r, c) is centred at 4 * (r, c) um, i.e. working pixel (2r, 2c)
    out = mesh_inputs.lumen_classes(lobules(tmp_path), "he", 1, slice(0, 3), slice(0, 3), 4.0)
    np.testing.assert_array_equal(out, WORKING[::2, ::2])


def test_window_offset(tmp_path: Path) -> None:
    out = mesh_inputs.lumen_classes(lobules(tmp_path), "he", 0, slice(2, 5), slice(1, 4), 2.0)
    np.testing.assert_array_equal(out, WORKING[2:5, 1:4])
    shifted = mesh_inputs.lumen_classes(lobules(tmp_path, origin=(2.0, 4.0)), "he", 0, slice(1, 4), slice(2, 5), 2.0)
    np.testing.assert_array_equal(shifted, WORKING[0:3, 0:3])  # working pixel (0, 0) is centred at (2, 4) um


def test_outside_working_grid_is_zero(tmp_path: Path) -> None:
    out = mesh_inputs.lumen_classes(lobules(tmp_path), "he", 0, slice(4, 9), slice(4, 9), 2.0)
    np.testing.assert_array_equal(out[:2, :2], WORKING[4:, 4:])
    assert not out[2:].any() and not out[:, 2:].any()
    assert out.shape == (5, 5)


def test_reference_mismatch(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="belongs to he, not other"):
        mesh_inputs.lumen_classes(lobules(tmp_path), "other", 0, slice(0, 6), slice(0, 6), 2.0)
