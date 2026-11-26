"""The zonation position and lumen classes that zonation_layers.py saves for the scene export."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("slideviz")
import zonation_layers

from slideviz.analysis import zonation

SIZE = 100
PEAKS = np.array([[20, 20], [70, 40]])
VALLEYS = np.array([[20, 80], [70, 60]])


def synthetic() -> tuple[zonation.Landscape, np.ndarray, np.ndarray]:
    """One working grid of 1 um pixels with two peaks and two valleys and three small holes, tissue and outline."""
    field = np.zeros((SIZE, SIZE), np.float32)
    for r, c in PEAKS:
        field[r, c] = 1.0
    for r, c in VALLEYS:
        field[r, c] = -1.0
    field = gaussian_filter(field, 6)
    inside = np.ones((SIZE, SIZE), bool)
    tissue = inside.copy()
    for r, c in ((20, 27), (20, 73), (70, 50)):  # beside the peak, beside the valley, between a peak and a valley
        tissue[r - 1: r + 2, c - 1: c + 2] = False
    tissue[45:48, 5:8] = False  # a hole beside neither
    land = zonation.Landscape(field, inside, inside, 1, 1.0, PEAKS, VALLEYS)
    return land, tissue, inside


def test_lumen_classes() -> None:
    land, tissue, inside = synthetic()
    port = zonation.portality(land, tissue, inside)
    assert port.lumen_class.dtype == np.uint8
    assert port.lumen_class[20, 27] == 2  # by the peak
    assert port.lumen_class[20, 73] == 1  # by the valley
    assert port.lumen_class[70, 50] == 3  # beside both
    assert port.lumen_class[46, 6] == 3  # a lumen without a vessel's peak or valley
    assert set(np.unique(port.lumen_class[tissue])) == {0}
    assert (port.lumen_class[~tissue] > 0).all()


def test_position_saved(tmp_path: Path) -> None:
    land, tissue, inside = synthetic()
    inside[:, :10] = False  # outside the outline
    port = zonation.portality(land, tissue, inside)
    path = tmp_path / "x_lobules.npz"
    zonation_layers.save_layers(path, land, port, inside, 1.8, 0.0, extra={"low": 0.5})
    with np.load(path) as saved:
        assert saved["position"].dtype == np.float32
        assert np.isnan(saved["position"][~inside]).all()
        np.testing.assert_array_equal(saved["position"][inside], port.points[inside])
        assert saved["lumen_class"].dtype == np.uint8
        np.testing.assert_array_equal(saved["lumen_class"], port.lumen_class)
        assert float(saved["working_um_per_px"]) == pytest.approx(1.8)
        np.testing.assert_array_equal(saved["working_origin_um"], [0.0, 0.0])
        assert float(saved["low"]) == 0.5
