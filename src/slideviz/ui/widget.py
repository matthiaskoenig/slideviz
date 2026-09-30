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

from slideviz.analysis.dab import CUTOFF, brownness_levels
from slideviz.analysis.dab import read_reference as read_dab_reference
from slideviz.analysis.masked import to_rgba
from slideviz.analysis.prediction import add_prediction_layers, tile_um_of
from slideviz.analysis.stain import normalise_levels
from slideviz.analysis.stain_stats import read as read_stain_reference
from slideviz.analysis.warp import load_field, nonrigid_levels
from slideviz.data.catalog import query, slide_path
from slideviz.data.registration import napari_affine
from slideviz.data.schema import Registration, Slide
from slideviz.io.reader import open_slide
from slideviz.settings import settings

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

# a block's name carries this prefix, a prediction file's animal does not
BLOCK_PREFIX = "apap_"

# one colour per stain, so an overlay reads as two channels instead of two pictures
STAIN_COLOURS = {"he": "green", "cyp2e1": "magenta", "cyp1a2": "magenta", "hmgb1": "cyan"}
FALLBACK_COLOUR = "yellow"

# how a stain is written in the layer list, where the index's lowercase key reads badly
STAIN_NAMES = {"he": "H&E", "cyp2e1": "CYP2E1", "cyp1a2": "CYP1A2", "hmgb1": "HMGB1"}

# separates the parts of a layer name: animal, then what it shows, then which kind
NAME_SEPARATOR = " · "

# what a prediction file is about, when it does not say; steatosis is the planned second
DEFAULT_LABEL = "necrosis"

# the DAB stains that carry a brownness readout, keyed as the index writes them
DAB_STAINS = ("cyp2e1", "cyp1a2")

# brownness is shown to twice the cutoff, so positive tissue spans the upper half
DAB_DISPLAY_MAX = CUTOFF * 2

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


def layer_name(animal: str, what: str, kind: str | None = None) -> str:
    """Return a layer name containing the animal, content, and optional kind."""
    parts = [animal, what] + ([kind] if kind else [])
    return NAME_SEPARATOR.join(parts)


def registration_lines(registration: Registration) -> list[str]:
    """Rigid method with its score, pose, and the non-rigid shift when there is one."""
    import numpy as np

    matrix = np.array(registration.matrix, float)
    rotation = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    scale = np.sqrt(abs(np.linalg.det(matrix[:2, :2])))
    method = registration.method
    for prefix, short in (("outline", "outline"), ("valis", "VALIS")):
        if method.startswith(prefix):
            method = short
    score = ""
    if registration.error_um is not None:
        score = f", {registration.error_um:.0f} um"
    elif registration.outline_dice is not None:
        score = f", Dice {registration.outline_dice:.3f}"
    lines = [f"rigid: {method}{score}", f"rotation {rotation:.1f} deg, scale {scale:.3f}"]
    nonrigid = registration.nonrigid
    if nonrigid is None:
        lines.append("non-rigid: none")
    else:
        lines.append(f"non-rigid: median {nonrigid.median_um:.0f} um, p95 {nonrigid.p95_um:.0f} um")
    return lines


class SlideList(QWidget):
    """Slide picker docked into the napari window."""

    def __init__(self, viewer, directory: Path | None = None) -> None:
        """Build and populate the slide picker.

        If provided, ``directory`` limits slides to that collection; otherwise,
        all indexed slides are shown.
        """
        super().__init__()
        self.viewer = viewer
        self.directory = str(directory.resolve()) if directory else None
        self._stain = None  # read on first use, then kept
        self._dab = None  # read on first use, then kept
        # level shapes of the block's reference slide, the grid a non-rigid layer is drawn on
        self._reference_shapes: list[tuple[int, int]] = []

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
        self.list.currentItemChanged.connect(self._show_info)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)

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
        layout.addWidget(self.info)
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
            item = QListWidgetItem(self._label(row))
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
    def _sidecar(row) -> Slide | None:
        """A slide's full sidecar, for the fields SQL does not carry."""
        sidecar = slide_path(row).with_suffix(".json")
        if not sidecar.exists():
            return None
        return Slide(**json.loads(sidecar.read_text()))

    @classmethod
    def _registration(cls, row) -> Registration | None:
        """The transform from a slide's sidecar."""
        slide = cls._sidecar(row)
        return slide.registration if slide is not None else None

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

    @staticmethod
    def _label(row) -> str:
        """One list entry per block: species, substance, dose, animal and stain count."""
        dose = row["dose_mg_per_kg"]
        return (
            f"{row['species']:<6} {row['substance']} "  # species first
            # xxx where the dose is not known yet, matching the placeholder in the filename
            f"{dose if dose is not None else 'xxx':>3} mg/kg  "
            f"{row['animal_id']:<3} "
            f"{row['n_slides']} stains"
        )

    def _show_info(self) -> None:
        """Fill the info panel with what the sidecars and predictions hold for a block."""
        selected = self._selected()
        if selected is None:
            self.info.setText("")
            return
        directory, block = selected
        slides = self._block_slides(directory, block)
        if not slides:
            self.info.setText("")
            return

        first = slides[0]
        dose = first["dose_mg_per_kg"]
        dose_text = dose if dose is not None else "xxx"
        title = f"{first['species']} {first['substance']}, {dose_text} mg/kg, {first['animal_id']}"
        lines = [f"<b>{title}</b>"]
        for row in slides:
            stain = STAIN_NAMES.get(row["stain"], row["stain"].upper())
            sidecar = self._sidecar(row)
            lines.append(f"<br><b>{stain}</b>")
            if sidecar is None:
                continue
            if sidecar.original_name:
                lines.append(f"file: {sidecar.original_name}")
            if sidecar.animal_id_corrected:
                lines.append(f"ID corrected: {sidecar.animal_id_corrected}")
            registration = sidecar.registration
            if registration is None:
                if row["stain"] != REFERENCE_STAIN:
                    lines.append("not registered")
                continue
            lines.extend(registration_lines(registration))

        prediction = self._prediction_file(block)
        if prediction is not None:
            try:
                record = json.loads(prediction.read_text())
                written = record.get("written", "")[:10]  # the model output's date
                lines.append(f"<br><b>Necrosis</b> (prediction file, {written})")
                lines.append(f"annotated {record['area_annotated']:.1%}, "
                             f"predicted {record['area_predicted']:.1%} of tissue")
            # a half-written prediction file should not take the panel down
            except (OSError, ValueError, KeyError) as exc:
                log.warning("unreadable prediction file %s: %s", prediction, exc)
        self.info.setText("<br>".join(lines))

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
            # applied always, so one slide has one appearance for everyone who opens it
            levels = self._matched(levels, Path(row["file"]).name.split(".")[0])
            stain = STAIN_NAMES.get(row["stain"], row["stain"].upper())
            # a scene number only means something on the files that hold more than one
            what = f"{stain} s{scene}" if info.n_scenes > 1 else stain
            name = layer_name(row["serial_block"].removeprefix(BLOCK_PREFIX), what)
            if registration is None:
                self._reference_shapes = [level.shape[:2] for level in levels]
            affine = None
            if registration is not None:
                affine = napari_affine(registration, reference_um or info.pixel_size_um)
            field = None
            if registration is not None and reference_um and self._reference_shapes:
                field = load_field(path.parent, registration)

            if field is None:
                self._add_stain(row, levels, name, info.pixel_size_um, affine)
                if row["stain"] in DAB_STAINS:
                    self._add_brownness(row, levels, affine, info.pixel_size_um)
            else:
                warped = nonrigid_levels(levels, self._reference_shapes, registration,
                                         field, reference_um)
                self._add_stain(row, warped, name, reference_um, None)
                if row["stain"] in DAB_STAINS:
                    self._add_brownness(row, warped, None, reference_um)
        # unreadable file, unsupported suffix, shape napari rejects; report, stay alive
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            log.exception("could not load %s", path)  # status line is transient, the log is not
            self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}")
            return None

        return info.pixel_size_um

    def _add_stain(self, row, levels: list, name: str, pixel_size_um: float, affine) -> None:
        """Add one stain pyramid as an RGB layer, placed by `affine` when given."""
        layer = self.viewer.add_image(
            self._levels_for(levels),
            name=name,
            rgb=True,
            multiscale=True,  # levels is a pyramid, napari picks one per zoom
            scale=(pixel_size_um, pixel_size_um),
            units="um",  # makes the scale bar read in micrometres
            affine=affine,
            colormap=STAIN_COLOURS.get(row["stain"], FALLBACK_COLOUR),
            opacity=0.7,
            blending="additive",  # so the stains show through each other
        )
        # keep the unmasked pyramid, so the toggle can swap the layer's data without opening the slide again
        layer.metadata[SOURCE_LEVELS] = levels

    def _dab_reference(self) -> tuple[float, dict] | None:
        """The brownness target and per-slide medians, read once and kept."""
        if self._dab is None:
            path = settings.dab_reference_file()
            if path is None:
                self._dab = ()
            else:
                try:
                    self._dab = read_dab_reference(path)
                    log.info("dab reference: %d slides from %s", len(self._dab[1]), path)
                except (OSError, ValueError, KeyError) as exc:
                    log.warning("unreadable dab reference %s: %s", path, exc)
                    self._dab = ()
        return self._dab or None

    def _add_brownness(self, row, levels: list, affine, pixel_size_um: float) -> None:
        """Add the DAB brownness map for a stain that carries one."""
        slide_key = Path(row["file"]).name.split(".")[0]
        reference = self._dab_reference()
        factor = 1.0
        if reference is not None:
            target, medians = reference
            median = medians.get(slide_key)
            # scaled onto the shared median, so one cutoff means the same on every slide
            factor = target / median if median else 1.0

        name = layer_name(
            row["serial_block"].removeprefix(BLOCK_PREFIX),
            STAIN_NAMES.get(row["stain"], row["stain"].upper()),
            "brownness",
        )
        self.viewer.add_image(
            [level[..., 0] for level in brownness_levels(levels, factor)],
            name=name,
            multiscale=True,
            scale=(pixel_size_um, pixel_size_um),
            units="um",
            affine=affine,
            colormap="inferno",
            contrast_limits=(0.0, DAB_DISPLAY_MAX),
            interpolation2d="linear",
            visible=False,  # the readout is a measurement, the stain is what one looks at
        )

    def _stain_reference(self) -> tuple[object, dict] | None:
        """The stain target and per-slide statistics, read once and kept."""
        if self._stain is None:
            path = settings.stain_reference_file()
            if path is None:
                self._stain = ()
            else:
                try:
                    self._stain = read_stain_reference(path)
                    log.info("stain reference: %d slides from %s", len(self._stain[1]), path)
                except (OSError, ValueError, KeyError) as exc:
                    log.warning("unreadable stain reference %s: %s", path, exc)
                    self._stain = ()
        return self._stain or None

    def _matched(self, levels: list, slide_key: str) -> list:
        """The pyramid matched onto the shared stain target, when the slide has statistics."""
        reference = self._stain_reference()
        if reference is None:
            return levels
        target, per_slide = reference
        stats = per_slide.get(slide_key)
        if stats is None:
            log.debug("no stain statistics for %s", slide_key)
            return levels
        return normalise_levels(levels, stats, target)

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

    @staticmethod
    def _prediction_file(block: str) -> Path | None:
        """The model output for one block, when a prediction directory is configured."""
        directory = settings.predictions_dir()
        if directory is None or not directory.is_dir():
            return None
        animal = block.removeprefix(BLOCK_PREFIX)
        # an unannotated slide writes its own name, so try both
        for suffix in ("_predictions.json", "_unlabelled_predictions.json"):
            path = directory / f"{animal}{suffix}"
            if path.exists():
                return path
        return None

    def _load_predictions(self, block: str, reference_um: float) -> int:
        """Add annotated and predicted tile maps in the reference stain's frame."""
        path = self._prediction_file(block)
        if path is None:
            return 0
        animal = block.removeprefix(BLOCK_PREFIX)
        try:
            record = json.loads(path.read_text())
            label = record.get("label", DEFAULT_LABEL)
            level_um = reference_um * 2 ** record["level"]
            layers = add_prediction_layers(
                self.viewer,
                path,
                tile_um_of(path, level_um),
                annotated_name=layer_name(animal, label, "annotated"),
                predicted_name=layer_name(animal, label, "predicted"),
            )
        # a truncated or half-written prediction file should not take the viewer down
        except (OSError, ValueError, KeyError) as exc:
            log.exception("could not load predictions from %s", path)
            self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}")
            return 0
        return len(layers)

    def _load_block(self, directory: str, block: str) -> None:
        """Add every stain of one block, aligned onto the reference where possible."""
        slides = self._block_slides(directory, block)
        if not slides:
            return

        reference_um, loaded, unaligned = None, 0, []
        self._reference_shapes = []  # another block's reference must not carry over
        for row in slides:
            pixel_size = self._load(row, reference_um)
            if pixel_size is None:
                continue
            loaded += 1
            if reference_um is None:  # the reference stain sorts first
                reference_um = pixel_size
            elif self._registration(row) is None:
                unaligned.append(row["stain"])

        maps = self._load_predictions(block, reference_um) if reference_um else 0

        note = f"  (overlaid, not registered: {', '.join(unaligned)})" if unaligned else ""
        maps_note = "  + necrosis maps" if maps else ""
        self.status.setText(f"{block}  {loaded} layers{maps_note}{note}")

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
