"""Resolve a slide stem to its file in any supported format."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("slideviz")
from slideviz.io.reader import find_slide, slide_paths, slide_stem


def test_slide_stem() -> None:
    assert slide_stem(Path("a/m1_he.ome.tiff")) == "m1_he"
    assert slide_stem(Path("a/m1_he.zarr")) == "m1_he"
    assert slide_stem(Path("a/M1_HE.CZI")) == "M1_HE"


def test_find_slide_any_format(tmp_path: Path) -> None:
    (tmp_path / "a_he.zarr").mkdir()
    (tmp_path / "b_he.ome.tiff").touch()
    assert find_slide(tmp_path, "a_he") == tmp_path / "a_he.zarr"
    assert find_slide(tmp_path, "b_he") == tmp_path / "b_he.ome.tiff"
    with pytest.raises(FileNotFoundError):
        find_slide(tmp_path, "c_he")


def test_find_slide_ignores_sidecars(tmp_path: Path) -> None:
    (tmp_path / "a_he.json").touch()
    with pytest.raises(FileNotFoundError):
        find_slide(tmp_path, "a_he")


def test_slide_paths_one_per_stem(tmp_path: Path) -> None:
    (tmp_path / "a_cyp2e1.zarr").mkdir()
    (tmp_path / "a_cyp2e1.json").touch()
    (tmp_path / "b_cyp2e1.ome.tiff").touch()
    (tmp_path / "a_he.zarr").mkdir()
    found = slide_paths(tmp_path, "*_cyp2e1")
    assert [p.name for p in found] == ["a_cyp2e1.zarr", "b_cyp2e1.ome.tiff"]
