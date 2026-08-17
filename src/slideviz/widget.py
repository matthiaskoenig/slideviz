"""napari dock widget listing the indexed slides."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
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

ORDER_SQL = "ORDER BY species, substance, dose_mg_per_kg, animal_id, stain"

# Filter label to the column it restricts
FILTERS = {"Species": "species", "Stain": "stain", "Dose": "dose_mg_per_kg"}

ANY = "All"


class SlideList(QWidget):
    """Slide picker docked into the napari window."""

    def __init__(self, viewer) -> None:
        """Build the list, the buttons and the status line, then fill the list."""
        super().__init__()
        self.viewer = viewer

        self.boxes = {}
        filters = QFormLayout()
        for label, column in FILTERS.items():
            box = QComboBox()
            box.currentTextChanged.connect(self.refresh)  # re-query on every change
            self.boxes[column] = box
            filters.addRow(label, box)

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
        layout.addLayout(filters)
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        layout.addWidget(self.status)

        self._fill_boxes()
        self.refresh()

    def _fill_boxes(self) -> None:
        """Offer the values the index actually holds, so new species appear on their own."""
        for column, box in self.boxes.items():
            values = query(f"SELECT DISTINCT {column} FROM slides ORDER BY 1")
            box.blockSignals(True)  # filling would otherwise fire refresh once per item
            box.clear()
            box.addItem(ANY)
            box.addItems([str(row[0]) for row in values])
            box.blockSignals(False)

    def _where(self) -> tuple[str, tuple]:
        """Build the WHERE clause from the active filters, and the values it needs."""
        clauses, params = [], []
        for column, box in self.boxes.items():
            if box.currentText() != ANY:
                clauses.append(f"{column} = ?")
                params.append(box.currentText())
        return ("WHERE " + " AND ".join(clauses) if clauses else "", tuple(params))

    def refresh(self) -> None:
        """Reload the list from the index, honouring the filters."""
        self.list.clear()
        where, params = self._where()
        rows = query(f"SELECT * FROM slides {where} {ORDER_SQL}", params)
        for row in rows:
            item = QListWidgetItem(self._label(row))
            item.setData(Qt.ItemDataRole.UserRole, str(slide_path(row)))  # hidden path
            self.list.addItem(item)
        self.status.setText(self._count())

    def _count(self) -> str:
        """Slides listed, and the total when a filter is hiding some."""
        total = query("SELECT COUNT(*) FROM slides")[0][0]
        shown = self.list.count()
        return f"{shown} slides" if shown == total else f"{shown} of {total} slides"

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
        self.status.setText(self._count())
