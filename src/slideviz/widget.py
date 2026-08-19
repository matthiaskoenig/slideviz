"""napari dock widget listing the indexed slides."""

from __future__ import annotations

import logging
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
from slideviz.reader import open_slide

# n_scenes rides along per row, so list can label scenes without opening any file
SELECT_SQL = "SELECT *, COUNT(*) OVER (PARTITION BY directory, file) AS n_scenes FROM slides"
ORDER_SQL = "ORDER BY species, substance, dose_mg_per_kg, animal_id, stain, scene"

# Filter label to the column it restricts
FILTERS = {"Species": "species", "Stain": "stain", "Dose": "dose_mg_per_kg"}

# The column's type, so a filter value is compared as what the column stores
FILTER_TYPES = {"species": str, "stain": str, "dose_mg_per_kg": int}

ANY = "All"

log = logging.getLogger(__name__)


def _column(name: str) -> str:
    """Check a column name before it goes into SQL, where it cannot be a parameter."""
    if name not in FILTERS.values():
        raise ValueError(f"not a filter column: {name}")
    return name


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

        self.reload()

    def _fill_boxes(self) -> None:
        """Offer the values the index actually holds, so new species appear on their own.

        Keeps each selection across a refill, so a reindex mid-session does not
        silently reset the filters.
        """
        for column, box in self.boxes.items():
            values = query(f"SELECT DISTINCT {_column(column)} FROM slides ORDER BY 1")
            previous = box.currentText()
            box.blockSignals(True)  # filling would otherwise fire refresh once per item
            box.clear()
            box.addItem(ANY)
            box.addItems([str(row[0]) for row in values])
            kept = box.findText(previous)  # -1 when the value is gone from the index
            box.setCurrentIndex(max(kept, 0))
            box.blockSignals(False)

    def _where(self) -> tuple[str, tuple]:
        """Build the WHERE clause from the active filters, and the values it needs."""
        clauses, params = [], []
        for column, box in self.boxes.items():
            if box.currentText() != ANY:
                clauses.append(f"{_column(column)} = ?")
                # cast, rather than leaving an integer column to SQLite's type affinity
                params.append(FILTER_TYPES[column](box.currentText()))
        return ("WHERE " + " AND ".join(clauses) if clauses else "", tuple(params))

    def reload(self) -> None:
        """Pick up a reindex: rebuild the dropdowns, then the list.

        refresh() alone keeps stale dropdowns, because they are filled once at
        construction and a new species would never appear in them.
        """
        self._fill_boxes()
        self.refresh()

    def refresh(self) -> None:
        """Reload the list from the index, honouring the filters."""
        self.list.clear()
        where, params = self._where()
        rows = query(f"{SELECT_SQL} {where} {ORDER_SQL}", params)
        for row in rows:
            item = QListWidgetItem(self._label(row))
            # path and scene ride on the item, so loading needs no second query
            item.setData(Qt.ItemDataRole.UserRole, (str(slide_path(row)), row["scene"]))
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
        # only multi-scene slides say which scene, so single-scene entries are unchanged
        scene = f"  scene {row['scene']}" if row["n_scenes"] > 1 else ""
        return (
            f"{row['species']:<6} {row['substance']} "  # species first
            f"{row['dose_mg_per_kg']:>3} mg/kg  "
            f"{row['animal_id']:<3} {row['stain']}{scene}"
        )

    def _selected(self) -> tuple[Path, int] | None:
        """Path and scene of the highlighted entry, or None when nothing is selected."""
        item = self.list.currentItem()
        if not item:
            return None
        path, scene = item.data(Qt.ItemDataRole.UserRole)
        return Path(path), scene

    def _load(self, path: Path, scene: int = 0) -> None:
        """Add one scene to the viewer as a multiscale layer (or report why not)."""
        try:
            info, levels = open_slide(path, scene)  # lazy, pixels arrive when napari draws
            name = f"{path.stem} s{scene}" if info.n_scenes > 1 else path.stem
            self.viewer.add_image(
                levels,
                name=name,
                rgb=True,
                multiscale=True,  # levels is a pyramid, napari picks one per zoom
                scale=(info.pixel_size_um, info.pixel_size_um),
                units="um",  # makes the scale bar read in micrometres
            )
        # unreadable file, unsupported suffix, shape napari rejects; report, stay alive
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            log.exception("could not load %s", path)  # status line is transient, the log is not
            self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}")
            return

        self.status.setText(f"{name}  {info.width}x{info.height} px")

    def _replace(self) -> None:
        """Drop the open layers and show the selected slide on its own."""
        selected = self._selected()
        if selected:
            self.viewer.layers.clear()
            self._load(*selected)

    def _add(self) -> None:
        """Show the selected slide alongside the ones already open."""
        selected = self._selected()
        if selected:
            before = len(self.viewer.layers)
            self._load(*selected)
            if len(self.viewer.layers) > before:  # nothing to say about a failed load
                # sections are separately mounted, so stacking is not alignment
                self.status.setText(f"{self.status.text()}  (overlaid, not registered)")

    def _clear(self) -> None:
        """Empty the viewer and reset the status line to the slide count."""
        self.viewer.layers.clear()
        self.status.setText(self._count())
