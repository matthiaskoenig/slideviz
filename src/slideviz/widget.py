"""napari dock widget listing the indexed slides."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from slideviz.catalog import query, slide_path
from slideviz.czi import read_pyramid

LIST_SQL = """
SELECT * FROM slides
ORDER BY species, substance, dose_mg_per_kg, animal_id, stain
"""


class SlideList(QWidget):
    """Slide picker docked into the napari window."""

    def __init__(self, viewer) -> None:
        """Build the list, the buttons and the status line, then fill the list."""
        super().__init__()
        self.viewer = viewer

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemDoubleClicked.connect(self._replace)  # second route to Load

        load = QPushButton("Load")
        load.clicked.connect(self._replace)
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)

        self.status = QLabel()

        buttons = QHBoxLayout()
        for button in (load, add, clear):
            buttons.addWidget(button)

        layout = QVBoxLayout(self)  # passing self installs it as this widget's layout
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        layout.addWidget(self.status)

        self.refresh()

    def refresh(self) -> None:
        """Reload the list from the index."""
        self.list.clear()
        rows = query(LIST_SQL)
        for row in rows:
            item = QListWidgetItem(self._label(row))
            item.setData(Qt.ItemDataRole.UserRole, str(slide_path(row)))  # hidden path
            self.list.addItem(item)
        self.status.setText(f"{len(rows)} slides")

    @staticmethod
    def _label(row) -> str:
        """One list entry, grouped so the two stains of an animal sit together."""
        return (
            f"{row['substance']} {row['dose_mg_per_kg']:>3} mg/kg  "  # padded to align
            f"{row['animal_id']:<3} {row['stain']}"
        )

    def _selected(self) -> Path | None:
        """Path of the highlighted entry, or None when nothing is selected."""
        item = self.list.currentItem()
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _load(self, path: Path) -> None:
        """Add one slide to the viewer as a multiscale layer."""
        info, levels = read_pyramid(path)  # lazy, pixels arrive when napari draws
        self.viewer.add_image(
            levels,
            name=path.stem,
            rgb=True,
            multiscale=True,  # levels is a pyramid, napari picks one per zoom
            scale=(info.pixel_size_um, info.pixel_size_um),
            units="um",  # makes the scale bar read in micrometres
        )
        self.status.setText(f"{path.stem}  {info.width}x{info.height} px")

    def _replace(self) -> None:
        """Drop the open layers and show the selected slide on its own."""
        path = self._selected()
        if path:
            self.viewer.layers.clear()
            self._load(path)

    def _add(self) -> None:
        """Show the selected slide alongside the ones already open."""
        path = self._selected()
        if path:
            self._load(path)

    def _clear(self) -> None:
        """Empty the viewer and reset the status line to the slide count."""
        self.viewer.layers.clear()
        self.status.setText(f"{self.list.count()} slides")
