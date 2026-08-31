"""napari dock widget listing the indexed slides."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
from slideviz.masked import to_rgba
from slideviz.reader import open_slide
from slideviz.registration import napari_affine
from slideviz.schema import Registration, Slide

# n_scenes rides along per row, so list can label scenes without opening any file
SELECT_SQL = "SELECT *, COUNT(*) OVER (PARTITION BY directory, file) AS n_scenes FROM slides"
ORDER_SQL = "ORDER BY species, substance, dose_mg_per_kg, animal_id, stain, scene"

# One row per serial block
BLOCK_SQL = """
SELECT directory, serial_block, species, substance, dose_mg_per_kg, animal_id,
       COUNT(*) AS n_slides
FROM slides
"""
BLOCK_GROUP_SQL = """
GROUP BY directory, serial_block
ORDER BY species, substance, dose_mg_per_kg, animal_id
"""

# the stain that every other one is registered to, so it goes in first and untransformed
REFERENCE_STAIN = "he"

# where a layer keeps its unmasked pyramid, so the background toggle can swap data
SOURCE_LEVELS = "slideviz_levels"

# one colour per stain, so an overlay reads as two channels instead of two pictures
STAIN_COLOURS = {"he": "green", "cyp2e1": "magenta", "cyp1a2": "magenta", "hmgb1": "cyan"}
FALLBACK_COLOUR = "yellow"

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

    def __init__(self, viewer, directory: Path | None = None) -> None:
        """Build the list, the buttons and the status line, then fill the list.

        A directory limits the list to that collection, so one index can hold
        several without them appearing as one. None lists everything indexed.
        """
        super().__init__()
        self.viewer = viewer
        self.directory = str(directory.resolve()) if directory else None

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

        self.hide_background = QCheckBox("Hide background")
        self.hide_background.setChecked(True)
        self.hide_background.setToolTip(
            "Make the white scan area and the staircase transparent, leaving tissue"
        )
        self.hide_background.toggled.connect(self._apply_background)

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
        layout.addWidget(self.hide_background)
        layout.addLayout(buttons)
        layout.addWidget(self.status)

        self.reload()

    def _fill_boxes(self) -> None:
        """Offer the values the index actually holds, so new species appear on their own.

        Keeps each selection across a refill, so a reindex mid-session does not
        silently reset the filters.
        """
        scope, params = self._scope()
        where = f"WHERE {scope[0]}" if scope else ""
        for column, box in self.boxes.items():
            # scoped too, or a filter would offer values no listed slide has
            values = query(
                f"SELECT DISTINCT {_column(column)} FROM slides {where} ORDER BY 1",
                tuple(params),
            )
            previous = box.currentText()
            box.blockSignals(True)  # filling would otherwise fire refresh once per item
            box.clear()
            box.addItem(ANY)
            box.addItems([str(row[0]) for row in values])
            kept = box.findText(previous)  # -1 when the value is gone from the index
            box.setCurrentIndex(max(kept, 0))
            box.blockSignals(False)

    def _scope(self) -> tuple[list[str], list]:
        """The collection clause, empty when the widget lists every directory."""
        if self.directory is None:
            return [], []
        return ["directory = ?"], [self.directory]

    def _where(self) -> tuple[str, tuple]:
        """Build the WHERE clause from the collection and the active filters."""
        clauses, params = self._scope()
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
        rows = query(f"{BLOCK_SQL} {where} {BLOCK_GROUP_SQL}", params)
        for row in rows:
            slides = self._block_slides(row["directory"], row["serial_block"])
            item = QListWidgetItem(self._label(row, slides))
            # the block key rides on the item, so loading needs no second lookup
            item.setData(
                Qt.ItemDataRole.UserRole, (row["directory"], row["serial_block"])
            )
            self.list.addItem(item)
        self.status.setText(self._count())

    @staticmethod
    def _block_slides(directory: str, block: str) -> list:
        """Every slide of one block, reference stain first so it is layer zero."""
        rows = query(
            f"{SELECT_SQL} WHERE directory = ? AND serial_block = ? {ORDER_SQL}",
            (directory, block),
        )
        return sorted(rows, key=lambda r: r["stain"] != REFERENCE_STAIN)

    @staticmethod
    def _registration(row) -> Registration | None:
        """The transform from a slide's sidecar, which SQL does not carry."""
        sidecar = slide_path(row).with_suffix(".json")
        if not sidecar.exists():
            return None
        slide = Slide(**json.loads(sidecar.read_text()))
        return slide.registration

    def _count(self) -> str:
        """Blocks listed, and the total when a filter is hiding some."""
        scope, params = self._scope()
        where = f"WHERE {scope[0]}" if scope else ""
        # the collection's total, not the index's, so the count matches the list
        total = query(
            f"SELECT COUNT(DISTINCT directory || serial_block) FROM slides {where}",
            tuple(params),
        )[0][0]
        shown = self.list.count()
        return f"{shown} blocks" if shown == total else f"{shown} of {total} blocks"

    def _label(self, row, slides) -> str:
        """One list entry per block, saying how well its stains are registered."""
        moving = [s for s in slides if s["stain"] != REFERENCE_STAIN]
        errors = [
            registration.error_um
            for s in moving
            if (registration := self._registration(s)) is not None
        ]
        if not errors:
            quality = "not registered"
        else:
            # the worst stain, since that is what limits the block as a whole
            known = [e for e in errors if e is not None]
            quality = f"{max(known):.0f} um" if known else "registered"
            if len(errors) < len(moving):
                quality += ", some not registered"

        return (
            f"{row['species']:<6} {row['substance']} "  # species first
            f"{row['dose_mg_per_kg']:>3} mg/kg  "
            f"{row['animal_id']:<3} "
            f"{row['n_slides']} stains  [{quality}]"
        )

    def _selected(self) -> tuple[str, str] | None:
        """Directory and block of the highlighted entry, or None when nothing is selected."""
        item = self.list.currentItem()
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _load(self, row, reference_um: float | None = None) -> float | None:
        """Add one slide as a multiscale layer, transformed when it has a transform."""
        path, scene = slide_path(row), row["scene"]
        registration = self._registration(row)
        try:
            info, levels = open_slide(path, scene)  # lazy, pixels arrive when napari draws
            name = f"{path.stem} s{scene}" if info.n_scenes > 1 else path.stem
            affine = None
            if registration is not None:
                affine = napari_affine(registration, reference_um or info.pixel_size_um)
            layer = self.viewer.add_image(
                self._levels_for(levels),
                name=name,
                rgb=True,
                multiscale=True,  # levels is a pyramid, napari picks one per zoom
                scale=(info.pixel_size_um, info.pixel_size_um),
                units="um",  # makes the scale bar read in micrometres
                affine=affine,
                colormap=STAIN_COLOURS.get(row["stain"], FALLBACK_COLOUR),
                opacity=0.7,
                blending="additive",  # so the stains show through each other
            )
            # keep the unmasked pyramid, so the toggle can swap the layer's data without opening the slide again
            layer.metadata[SOURCE_LEVELS] = levels
        # unreadable file, unsupported suffix, shape napari rejects; report, stay alive
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            log.exception("could not load %s", path)  # status line is transient, the log is not
            self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}")
            return None

        return info.pixel_size_um

    def _levels_for(self, levels: list) -> list:
        """The pyramid as the checkbox currently wants it: masked or untouched."""
        if not self.hide_background.isChecked():
            return levels
        # stays lazy: an alpha chunk is built only for the level being drawn
        return to_rgba(levels)

    def _apply_background(self) -> None:
        """Re-mask every open layer, so the checkbox acts on what is already shown."""
        for layer in self.viewer.layers:
            levels = layer.metadata.get(SOURCE_LEVELS)
            if levels is None:  # a layer this widget did not load
                continue
            layer.data = self._levels_for(levels)

    def _load_block(self, directory: str, block: str) -> None:
        """Add every stain of one block, aligned onto the reference where possible."""
        slides = self._block_slides(directory, block)
        if not slides:
            return

        reference_um, loaded, unaligned = None, 0, []
        for row in slides:
            pixel_size = self._load(row, reference_um)
            if pixel_size is None:
                continue
            loaded += 1
            if reference_um is None:  # the reference stain sorts first
                reference_um = pixel_size
            elif self._registration(row) is None:
                unaligned.append(row["stain"])

        note = f"  (overlaid, not registered: {', '.join(unaligned)})" if unaligned else ""
        self.status.setText(f"{block}  {loaded} layers{note}")

    def _replace(self) -> None:
        """Drop the open layers and show the selected block on its own."""
        selected = self._selected()
        if selected:
            self.viewer.layers.clear()
            self._load_block(*selected)

    def _add(self) -> None:
        """Show the selected block alongside the blocks already open."""
        selected = self._selected()
        if selected:
            self._load_block(*selected)

    def _clear(self) -> None:
        """Empty the viewer and reset the status line to the slide count."""
        self.viewer.layers.clear()
        self.status.setText(self._count())
