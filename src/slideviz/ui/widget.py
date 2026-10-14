"""napari dock widget listing the indexed slides."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from napari.qt.threading import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from slideviz.analysis.dab import STEPS, gradient_steps
from slideviz.analysis.flatten import flatten_levels
from slideviz.analysis.flatten import load as load_flattening
from slideviz.analysis.masked import to_rgba
from slideviz.analysis.prediction import add_prediction_layers, read_lumen, tile_um_of
from slideviz.analysis.stain import normalise_levels
from slideviz.analysis.stain_stats import read as read_stain_reference
from slideviz.analysis.tissue import mask_from_level, pick_level
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

# where a layer keeps its unmasked pyramids, raw and evened, so the toggles can swap data
SOURCE_LEVELS = "slideviz_levels"
EVENED_LEVELS = "slideviz_evened_levels"

# a slide's flattening fields sit in this folder beside it, as <stem>_flatten.npz
FLATTEN_DIR = "flatten"

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

# the DAB stains that carry a gradient readout, keyed as the index writes them
DAB_STAINS = ("cyp2e1", "cyp1a2")

# stored gradient map of a DAB slide, from scripts/cyp2e1_readout.py
READOUT_SUFFIX = "_readout.npz"

# that run's per-slide numbers, beside the maps
READOUT_RECORD = "cyp2e1_readout.json"

# necrosis_lumen.py output beside the predictions: one lumen file per animal and the run's areas
LUMEN_SUFFIX = "_lumen.npz"
LUMEN_RECORD = "necrosis_lumen.json"

# the share of necrotic tissue that is DAB positive is shown from this much necrosis on, as reported
NECROTIC_SHOWN_PCT = 5.0

# gradient steps 1 to 4 from olive to gold, rising in strength; step 0 and outside the tissue clear.
# Gold stays gold over pink H&E, where brown turns brick red, and under the cyan necrosis it mixes to green.
STEP_RGB = ((200, 196, 72), (218, 213, 48), (237, 229, 24), (255, 246, 0))
STEP_ALPHA = (0.56, 0.74, 0.87, 1.0)  # times STEP_OPACITY, so at most half covers the slide
STEP_OPACITY = 0.5
STEP_COLOURS = Colormap(
    [[0.0, 0.0, 0.0, 0.0]]
    + [[c / 255 for c in rgb] + [alpha] for rgb, alpha in zip(STEP_RGB, STEP_ALPHA, strict=True)]
    + [[0.0, 0.0, 0.0, 0.0]],
    controls=[0.0] + [(i + 0.5) / STEPS for i in range(STEPS)] + [1.0],
    interpolation="zero",
    name="dab gradient",
)

# Filter label to the column it restricts
FILTERS = {"Species": "species", "Stain": "stain", "Dose": "dose_mg_per_kg"}

# The column's type, so a filter value is compared as what the column stores
FILTER_TYPES = {"species": str, "stain": str, "dose_mg_per_kg": int}

ANY = "All"

PIXEL_POLL_MS = 200

log = logging.getLogger(__name__)


def _column(name: str) -> str:
    """Check a column name before it goes into SQL, where it cannot be a parameter."""
    if name not in FILTERS.values():
        raise ValueError(f"not a filter column: {name}")
    return name


def block_title(row) -> str:
    """A block as species, substance, dose and animal, e.g. mouse apap, 500 mg/kg, m1."""
    dose = row["dose_mg_per_kg"]
    # xxx where the dose is not known yet, matching the placeholder in the filename
    return f"{row['species']} {row['substance']}, {dose if dose is not None else 'xxx'} mg/kg, {row['animal_id']}"


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
        self._readout = None  # the readout run's numbers, read on first use, then kept
        self._lumen_areas = None  # necrotic areas without lumina per animal, read on first use, then kept
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
        self.hide_background.toggled.connect(self._apply_display)

        self.even_staining = QCheckBox("Even out staining")
        self.even_staining.setChecked(True)
        self.even_staining.setToolTip(
            "Divide out slide-wide staining gradients for display; measurements use the raw pixels"
        )
        self.even_staining.toggled.connect(self._apply_display)

        load = QPushButton("Load")
        load.clicked.connect(self._replace)
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)

        self.status = QLabel()
        self.activity = QLabel()
        for label in (self.status, self.activity):
            label.setWordWrap(True)  # long messages wrap instead of widening the dock
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self._show_activity(None)
        self._buttons = (load, add, clear)
        self._worker = None  # the block being opened, one at a time

        buttons = QHBoxLayout()
        for button in (load, add, clear):
            buttons.addWidget(button)

        layout = QVBoxLayout(self)  # passing self installs it as this widget's layout
        layout.addLayout(filters)
        layout.addWidget(self.list)
        layout.addWidget(self.info)
        layout.addWidget(self.hide_background)
        layout.addWidget(self.even_staining)
        layout.addLayout(buttons)
        layout.addWidget(self.activity)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self._pixels = QTimer(self)
        self._pixels.timeout.connect(self._poll_pixels)
        self._pixels.start(PIXEL_POLL_MS)

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
        path = slide_path(row)
        # <stem>.ome.json locally, <stem>.json on the NAS
        for sidecar in (path.with_suffix(".json"), path.with_name(f"{path.name.split('.')[0]}.json")):
            if sidecar.exists():
                return Slide(**json.loads(sidecar.read_text()))
        return None

    @classmethod
    def _registration(cls, row) -> Registration | None:
        """The transform from a slide's sidecar."""
        slide = cls._sidecar(row)
        return slide.registration if slide is not None else None

    @staticmethod
    def _label(row) -> str:
        """One list entry per block: species, substance, dose and animal."""
        dose = row["dose_mg_per_kg"]
        return (
            f"{row['species']:<6} {row['substance']} "  # species first
            # xxx where the dose is not known yet, matching the placeholder in the filename
            f"{dose if dose is not None else 'xxx':>3} mg/kg  "
            f"{row['animal_id']}"
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

        lines = [f"<b>{block_title(slides[0])}</b>"]
        for row in slides:
            sidecar = self._sidecar(row)
            if sidecar is None:
                continue
            details = [f"ID corrected: {sidecar.animal_id_corrected}"] if sidecar.animal_id_corrected else []
            if sidecar.registration is not None:
                details.extend(registration_lines(sidecar.registration))
            elif row["stain"] != REFERENCE_STAIN:
                details.append("not registered")
            # a stain gets a heading only when something is said about it
            if details:
                lines.append(f"<br><b>{STAIN_NAMES.get(row['stain'], row['stain'].upper())}</b>")
                lines.extend(details)

        prediction = self._prediction_file(block)
        if prediction is not None:
            try:
                record = json.loads(prediction.read_text())
                written = record.get("written", "")[:10]  # the model output's date
                lines.append(f"<br><b>Necrosis</b> (prediction file, {written})")
                areas = self._lumen_record().get(block.removeprefix(BLOCK_PREFIX))
                if areas is not None:
                    lines.append(f"annotated {areas['area_annotated_no_lumen']:.1%}, "
                                 f"predicted {areas['area_predicted_no_lumen']:.1%} of tissue without lumina")
                else:
                    lines.append(f"annotated {record['area_annotated']:.1%}, "
                                 f"predicted {record['area_predicted']:.1%} of tissue")
            # a half-written prediction file should not take the panel down
            except (OSError, ValueError, KeyError) as exc:
                log.warning("unreadable prediction file %s: %s", prediction, exc)
        lines.extend(self._readout_lines(slides))
        self.info.setText("<br>".join(lines))

    def _readout_record(self) -> dict:
        """The CYP2E1 readout run's record from the prediction directory, read once and kept."""
        if self._readout is None:
            self._readout = {}
            directory = settings.predictions_dir()
            path = directory / READOUT_RECORD if directory is not None else None
            if path is not None and path.exists():
                try:
                    self._readout = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    log.warning("unreadable readout record %s: %s", path, exc)
        return self._readout

    def _lumen_record(self) -> dict:
        """Necrotic areas without lumina per animal, from the lumen run's record, read once and kept."""
        if self._lumen_areas is None:
            self._lumen_areas = {}
            directory = settings.predictions_dir()
            path = directory / LUMEN_RECORD if directory is not None else None
            if path is not None and path.exists():
                try:
                    rows = json.loads(path.read_text())["slides_areas"]
                    self._lumen_areas = {row["animal"]: row for row in rows}
                except (OSError, ValueError, KeyError) as exc:
                    log.warning("unreadable lumen record %s: %s", path, exc)
        return self._lumen_areas

    def _readout_lines(self, slides: list) -> list[str]:
        """Info lines with the CYP2E1-positive share of each DAB slide in the block."""
        record = self._readout_record()
        measured = record.get("slides_measured", {})
        lines = []
        for row in slides:
            found = measured.get(Path(row["file"]).name.split(".")[0])
            if row["stain"] not in DAB_STAINS or found is None:
                continue
            stain = STAIN_NAMES.get(row["stain"], row["stain"].upper())
            lines.append(f"<br><b>{stain} positive</b> (readout, {record.get('written', '')[:10]})")
            if found.get("positive_surviving_pct") is not None:
                lines.append(f"{found['positive_surviving_pct']:.1f}% of surviving tissue")
            if (found.get("positive_necrotic_pct") is not None
                    and found.get("necrotic_pct", 0) >= NECROTIC_SHOWN_PCT):
                lines.append(f"{found['positive_necrotic_pct']:.1f}% of necrotic tissue")
            lines.append(f"{found['positive_pct']:.1f}% of all tissue")
        return lines

    def _selected(self) -> tuple[str, str] | None:
        """Directory and block of the highlighted entry, or None when nothing is selected."""
        item = self.list.currentItem()
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _open(self, row, reference_um: float | None = None) -> dict:
        """Open one slide and build its lazy layer data; touches no widget."""
        path, scene = slide_path(row), row["scene"]
        registration = self._registration(row)
        info, levels = open_slide(path, scene)  # lazy, pixels arrive when napari draws
        slide_key = Path(row["file"]).name.split(".")[0]
        pyramids = {SOURCE_LEVELS: levels}
        # fields are fitted on a file's only scene, on its own pixel grid, so they go in before the warp
        flat = self._flattening(path.parent, slide_key) if info.n_scenes == 1 else None
        if flat is not None:
            pyramids[EVENED_LEVELS] = flatten_levels(levels, flat, info.pixel_size_um)
        # applied always, so one slide has one appearance for everyone who opens it
        pyramids = {key: self._matched(p, slide_key) for key, p in pyramids.items()}
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

        shown_um = info.pixel_size_um
        if field is not None:
            pyramids = {key: nonrigid_levels(p, self._reference_shapes, registration, field, reference_um)
                        for key, p in pyramids.items()}
            shown_um, affine = reference_um, None
        return {"row": row, "pyramids": pyramids, "name": name, "shown_um": shown_um,
                "affine": affine, "pixel_size_um": info.pixel_size_um,
                "registered": registration is not None}

    @staticmethod
    def _flattening(directory: Path, slide_key: str):
        """A slide's flattening fields from the folder beside it, or None when absent or unreadable."""
        path = directory / FLATTEN_DIR / f"{slide_key}_flatten.npz"
        if not path.exists():
            log.debug("no flattening fields for %s", slide_key)
            return None
        try:
            return load_flattening(path)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("unreadable flattening fields %s: %s", path, exc)
            return None

    def _show(self, opened: dict) -> bool:
        """Add an opened slide's layers; False when napari rejects them."""
        row = opened["row"]
        try:
            self._add_stain(row, opened["pyramids"], opened["name"], opened["shown_um"],
                            opened["affine"])
        except (RuntimeError, ValueError, KeyError) as exc:
            log.exception("could not show %s", opened["name"])
            self.status.setText(f"{opened['name']}: {type(exc).__name__}: {exc}")
            return False
        return True

    def _add_stain(self, row, pyramids: dict, name: str, pixel_size_um: float, affine) -> None:
        """Add one stain as an RGB layer, placed by `affine` when given."""
        self.viewer.add_image(
            self._levels_for(pyramids),
            name=name,
            rgb=True,
            multiscale=True,  # levels is a pyramid, napari picks one per zoom
            scale=(pixel_size_um, pixel_size_um),
            units="um",  # makes the scale bar read in micrometres
            affine=affine,
            colormap=STAIN_COLOURS.get(row["stain"], FALLBACK_COLOUR),
            opacity=0.7,
            blending="additive",  # so the stains show through each other
            # keep the unmasked pyramids, so the toggles can swap the layer's data without opening the slide again
            metadata=dict(pyramids),
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
                    log.debug("stain reference: %d slides from %s", len(self._stain[1]), path)
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

    def _levels_for(self, pyramids: dict) -> list:
        """The pyramid as the checkboxes currently want it: evened or raw, masked or untouched."""
        levels = pyramids[SOURCE_LEVELS]
        if self.even_staining.isChecked() and EVENED_LEVELS in pyramids:
            levels = pyramids[EVENED_LEVELS]
        if not self.hide_background.isChecked():
            return levels
        import numpy as np

        # the mask comes from the uncorrected pixels: evening out pales the stained edge, which Otsu then drops
        source = pyramids[SOURCE_LEVELS]
        mask = mask_from_level(np.asarray(source[pick_level(source)]))
        # stays lazy: an alpha chunk is built only for the level being drawn
        return to_rgba(levels, mask)

    def _apply_display(self) -> None:
        """Re-pick every open layer's pyramid, so the checkboxes act on what is already shown."""
        for layer in self.viewer.layers:
            if SOURCE_LEVELS not in layer.metadata:  # a layer this widget did not load
                continue
            layer.data = self._levels_for(layer.metadata)

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
        """Add the predicted necrotic area in the reference stain's frame."""
        path = self._prediction_file(block)
        if path is None:
            return 0
        animal = block.removeprefix(BLOCK_PREFIX)
        try:
            record = json.loads(path.read_text())
            label = record.get("label", DEFAULT_LABEL)
            level_um = reference_um * 2 ** record["level"]
            try:
                lumen = read_lumen(path.parent / f"{animal}{LUMEN_SUFFIX}")
            # a broken lumen file costs the cut-out only, the maps still load
            except (OSError, ValueError, KeyError) as exc:
                log.warning("unreadable lumen file for %s: %s", animal, exc)
                lumen = None
            if lumen is None:
                log.info("no lumen file for %s, necrosis maps keep the lumina", animal)
            layers = add_prediction_layers(
                self.viewer,
                path,
                tile_um_of(path, level_um),
                predicted_name=layer_name(animal, label, "predicted"),
                lumen=lumen,
            )
        # a truncated or half-written prediction file should not take the viewer down
        except (OSError, ValueError, KeyError) as exc:
            log.exception("could not load predictions from %s", path)
            self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}")
            return 0
        return len(layers)

    def _load_readouts(self, slides: list) -> int:
        """Add the stored gradient map of each DAB slide, on its H&E partner's grid."""
        import numpy as np

        directory = settings.predictions_dir()
        if directory is None or not directory.is_dir():
            return 0
        references = {Path(row["file"]).name.split(".")[0]
                      for row in slides if row["stain"] == REFERENCE_STAIN}
        added = 0
        for row in slides:
            if row["stain"] not in DAB_STAINS:
                continue
            path = directory / f"{Path(row['file']).name.split('.')[0]}{READOUT_SUFFIX}"
            if not path.exists():
                continue
            try:
                with np.load(path) as stored:
                    steps = gradient_steps(stored["position"])
                    pixel_um = float(stored["um_per_px"])
                    reference = str(stored["reference"])
            except (OSError, ValueError, KeyError) as exc:
                log.exception("could not load readout %s", path)
                self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            # the map is drawn in the H&E frame, so it only fits that H&E slide
            if reference not in references:
                log.warning("%s is on %s, not on this block's H&E", path.name, reference)
                continue
            self.viewer.add_image(
                steps,
                name=layer_name(row["serial_block"].removeprefix(BLOCK_PREFIX),
                                STAIN_NAMES.get(row["stain"], row["stain"].upper()), "gradient"),
                scale=(pixel_um, pixel_um),
                translate=(pixel_um / 2, pixel_um / 2),  # napari centres pixels on their indices
                units="um",
                colormap=STEP_COLOURS,
                opacity=STEP_OPACITY,
                contrast_limits=(0, STEPS),
                interpolation2d="nearest",  # steps are classes, a blend between two would be a third
                visible=False,
            )
            added += 1
        return added

    @thread_worker
    def _open_block(self, slides: list):
        """Open a block's slides in order, yielding (row, opened or error) for each."""
        reference_um = None
        for row in slides:
            try:
                opened = self._open(row, reference_um)
            # unreadable file, unsupported suffix, missing sidecar field
            except (RuntimeError, ValueError, OSError, KeyError) as exc:
                log.exception("could not open %s", slide_path(row))
                yield row, exc
                continue
            if reference_um is None:  # the reference stain sorts first
                reference_um = opened["pixel_size_um"]
            yield row, opened

    def _load_block(self, directory: str, block: str) -> None:
        """Open every stain of one block off the UI thread, adding each as it is ready."""
        slides = self._block_slides(directory, block)
        if not slides or self._worker is not None:
            return

        animal = block.removeprefix(BLOCK_PREFIX)
        state = {"reference_um": None, "unaligned": [], "done": 0}
        self._reference_shapes = []  # another block's reference must not carry over
        self.status.setText("")  # the status line only carries this load's problems
        self._set_busy(True)
        self._step(animal, slides[0], 0, len(slides))

        def arrived(item) -> None:
            """Add one opened slide and move the bar on."""
            row, opened = item
            state["done"] += 1
            if isinstance(opened, Exception):
                self.status.setText(f"{slide_path(row).name}: {type(opened).__name__}: {opened}")
            elif self._show(opened):
                if state["reference_um"] is None:
                    state["reference_um"] = opened["pixel_size_um"]
                elif not opened["registered"]:
                    state["unaligned"].append(row["stain"])
            if state["done"] < len(slides):
                self._step(animal, slides[state["done"]], state["done"], len(slides))

        def finished() -> None:
            """Add the prediction maps and warn about slides shown without registration."""
            self._worker = None
            self._set_busy(False)
            reference_um = state["reference_um"]
            # CYP2E1 first, so the see-through necrosis lies on top and their overlap mixes
            self._load_readouts(slides)
            if reference_um:
                self._load_predictions(block, reference_um)
            log.info("opened %s", block_title(slides[0]))
            if state["unaligned"]:
                self.status.setText(f"{block}: overlaid, not registered: {', '.join(state['unaligned'])}")

        self._worker = self._open_block(slides)
        self._worker.yielded.connect(arrived)
        self._worker.errored.connect(
            lambda exc: self.status.setText(f"{block}: {type(exc).__name__}: {exc}"))
        self._worker.finished.connect(finished)  # emitted after an error too
        self._worker.start()

    def _step(self, animal: str, row, done: int, total: int) -> None:
        """Show which slide is being opened."""
        stain = STAIN_NAMES.get(row["stain"], row["stain"].upper())
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self._show_activity(f"{animal}: opening {stain} ({done + 1}/{total})")

    def _set_busy(self, busy: bool) -> None:
        """Lock the controls while a block opens."""
        for control in (*self._buttons, self.list):
            control.setEnabled(not busy)
        if not busy:
            self._show_activity(None)

    def _poll_pixels(self) -> None:
        """Show a busy bar while napari is still fetching pixels for any layer."""
        if self._worker is not None:
            return
        waiting = [layer for layer in self.viewer.layers if not layer.loaded]
        if not waiting:
            self._show_activity(None)
            return
        self.progress.setRange(0, 0)  # busy indicator, the tile count is unknown
        self._show_activity(f"loading pixels: {len(waiting)} of {len(self.viewer.layers)} layers")

    def _show_activity(self, text: str | None) -> None:
        """Show the activity line and bar with `text`, or hide both for None."""
        self.activity.setVisible(text is not None)
        self.progress.setVisible(text is not None)
        if text is not None:
            self.activity.setText(text)

    def _replace(self) -> None:
        """Drop the open layers and show the selected block on its own."""
        selected = self._selected()
        if selected and self._worker is None:
            self.viewer.layers.clear()
            self._load_block(*selected)

    def _add(self) -> None:
        """Show the selected block alongside the blocks already open."""
        selected = self._selected()
        if selected:
            self._load_block(*selected)

    def _clear(self) -> None:
        """Empty the viewer and the status line."""
        self.viewer.layers.clear()
        self.status.setText("")
