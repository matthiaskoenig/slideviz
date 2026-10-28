"""napari dock widget listing the indexed slides."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from napari.layers import Surface, Vectors
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
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from slideviz.analysis.dab import (
    POSITION_SCALE,
    STEP_ALPHA,
    STEP_OPACITY,
    STEP_RGB,
    STEPS,
    gradient_steps,
)
from slideviz.analysis.flatten import DAB_STAINS, FLATTEN_DIR, flatten_levels
from slideviz.analysis.flatten import load as load_flattening
from slideviz.analysis.masked import to_rgba
from slideviz.analysis.prediction import (
    CALL_LOOK,
    NECROSIS_RECORD,
    add_call_layer,
    call_level,
)
from slideviz.analysis.stain import normalise_levels
from slideviz.analysis.stain_stats import read as read_stain_reference
from slideviz.analysis.tissue import mask_from_level, pick_level
from slideviz.analysis.warp import load_field, nonrigid_levels
from slideviz.data.catalog import query, slide_path
from slideviz.data.registration import find_sidecar, napari_affine
from slideviz.data.schema import Registration, Slide
from slideviz.io.reader import open_slide
from slideviz.settings import settings

SELECT_SQL = "SELECT * FROM slides"
ORDER_SQL = "ORDER BY species, substance, dose_mg_per_kg, animal_id, stain, scene"

# One row per serial block
BLOCK_SQL = """
SELECT directory, serial_block, species, substance, dose_mg_per_kg, animal_id
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

# how a stain is written in layer names, the info panel and the activity line
STAIN_NAMES = {"he": "H&E", "cyp2e1": "CYP2E1", "cyp1a2": "CYP1A2", "hmgb1": "HMGB1"}

# where a layer keeps its own placement (its registration, or none), so turning the view composes on top of it
BASE_AFFINE = "slideviz_base_affine"
TURN_STEP = 90  # degrees per press of a turn button

# separates the parts of a layer name: animal, then what it shows
NAME_SEPARATOR = " · "

# what the necrosis map's layer is called
NECROSIS_NAME = "necrosis"

# stored gradient map of a DAB slide, from scripts/cyp2e1_readout.py
READOUT_SUFFIX = "_readout.npz"

# that run's per-slide numbers, beside the maps; a rerun is added under a dated name, and the newest is shown
READOUT_RECORDS = "cyp2e1_readout*.json"

# necrosis_slides.py output beside the predictions: the necrosis map per animal and the run's areas
NECROSIS_SUFFIX = "_necrosis.npz"

# the share of necrotic tissue that is DAB positive is shown from this much necrosis on, as reported
NECROTIC_SHOWN_PCT = 5.0

# the gradient steps' colours (analysis/dab.py) as a napari colormap, one colour per step
STEP_COLOURS = Colormap(
    [[0.0, 0.0, 0.0, 0.0]]
    + [[c / 255 for c in rgb] + [alpha] for rgb, alpha in zip(STEP_RGB, STEP_ALPHA, strict=True)]
    + [[0.0, 0.0, 0.0, 0.0]],
    controls=[0.0] + [(i + 0.5) / STEPS for i in range(STEPS)] + [1.0],
    interpolation="zero",
    name="dab gradient",
)

# mesh/fields.py output in the mesh directory, per mesh: the corners, triangles and values, with its record (.json)
# and the build_mesh.py view file (named edges) beside it
MESH_FIELDS = "_fields.npz"
MESH_VIEW = "_view.npz"

# CYP2E1 on a mesh, clear at the low end to the top step's gold; vispy colours each corner and blends the colours
# across a triangle, which equals the colour of the blended value on a straight ramp
MESH_CYP2E1 = Colormap(
    [[*(c / 255 for c in STEP_RGB[0]), 0.0], [*(c / 255 for c in STEP_RGB[-1]), 1.0]],
    name="mesh cyp2e1",
)
MESH_LINE_RGB = (90 / 255, 90 / 255, 90 / 255)  # triangle sides, as in the mesh pictures
MESH_LINE_PX = 1  # the wireframe width is in screen pixels

# build_mesh.py edge tags: vessel wall, section edge, cut side, necrosis border, coloured as in the mesh pictures
MESH_EDGE_RGB = {11: (30, 90, 220), 12: (240, 140, 0), 13: (60, 160, 60), 14: (0, 0, 0)}
# napari draws a vector's width in data units (um), so the width is reset to this many screen pixels on every zoom
MESH_EDGE_PX = 2
MESH_EDGES = "slideviz_mesh_edges"  # metadata key of a mesh's named edge layer

# a flat view needs no depth test; with one, layers on the same plane can hide each other's lines
MESH_BLENDING = "translucent_no_depth"

# Filter label to the column it restricts
FILTERS = {"Species": "species", "Stain": "stain", "Dose": "dose_mg_per_kg"}

# The column's type, so a filter value is compared as what the column stores
FILTER_TYPES = {"species": str, "stain": str, "dose_mg_per_kg": int}

ANY = "All"

PIXEL_POLL_MS = 200

log = logging.getLogger(__name__)


def slide_stem(row) -> str:
    """A slide's file name up to the first dot, the key its readout, flattening and stain statistics share."""
    return Path(row["file"]).name.split(".")[0]


def stain_name(stain: str) -> str:
    """How a stain is written in layer names and the info panel."""
    return STAIN_NAMES.get(stain, stain.upper())


def _column(name: str) -> str:
    """The column name, checked against the filter columns before it is written into SQL text."""
    if name not in FILTERS.values():
        raise ValueError(f"not a filter column: {name}")
    return name


def block_title(row) -> str:
    """A block as species, substance, dose and animal, e.g. mouse apap, 500 mg/kg, m1; rat, mdo26 without the two."""
    dose = row["dose_mg_per_kg"]
    head = " ".join(part for part in (row["species"], row["substance"]) if part)
    return ", ".join([head, *([f"{dose} mg/kg"] if dose is not None else []), row["animal_id"]])


def turned(angle: float):
    """The 2D affine that turns the view by `angle` degrees clockwise on screen about the world origin."""
    import numpy as np

    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    # world axes are (row, col) with rows running down the screen
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def animal_name(block: str, substance: str | None) -> str:
    """A block's animal as prediction files name it: the block minus its substance prefix, apap_281mg_m1 to 281mg_m1."""
    return block.removeprefix(f"{substance}_") if substance else block


def layer_name(animal: str, what: str) -> str:
    """A layer name from the animal and what the layer shows: a stain, or a readout such as necrosis."""
    return NAME_SEPARATOR.join([animal, what])


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
        parts = [f"{label} {value:.0f} um" for label, value in
                 (("median", nonrigid.median_um), ("p95", nonrigid.p95_um)) if value is not None]
        lines.append("non-rigid: " + (", ".join(parts) or nonrigid.method))
    return lines


def mesh_layers(fields_path: Path, animal: str) -> list:
    """One stored mesh as hidden layers: CYP2E1 at the corners with the triangle sides, necrotic triangles, named edges."""
    import numpy as np

    record = json.loads(fields_path.with_suffix(".json").read_text())
    size = record["mesh"]["size_um"]
    mesh = f"mesh {record['mesh']['variant']} {size['small']:g}-{size['large']:g} um"
    with np.load(fields_path) as stored:
        corners = stored["corner_xy_um"][:, ::-1]  # (x, y) to napari's (row, col), um in the H&E frame
        triangles, cyp2e1, necrotic = stored["triangle_corners"], stored["cyp2e1_corners"], stored["necrotic"]
    with np.load(fields_path.with_name(fields_path.name.removesuffix(MESH_FIELDS) + MESH_VIEW)) as stored:
        points, edges, tags = stored["points_um"][:, ::-1], stored["edges"], stored["edge_tag"]
    shown = {"units": "um", "blending": MESH_BLENDING, "visible": False}
    return [
        Surface((corners, triangles, cyp2e1), name=layer_name(animal, NAME_SEPARATOR.join([mesh, "CYP2E1"])),
                colormap=MESH_CYP2E1, contrast_limits=(0, POSITION_SCALE), opacity=STEP_OPACITY,
                wireframe={"visible": True, "color": MESH_LINE_RGB, "width": MESH_LINE_PX}, **shown),
        Surface((corners, triangles[necrotic], np.ones(len(corners))),
                name=layer_name(animal, NAME_SEPARATOR.join([mesh, NECROSIS_NAME])),
                colormap=CALL_LOOK["colormap"], contrast_limits=CALL_LOOK["contrast_limits"],
                opacity=CALL_LOOK["opacity"], **shown),
        Vectors(np.stack([points[edges[:, 0]], points[edges[:, 1]] - points[edges[:, 0]]], axis=1),
                name=layer_name(animal, NAME_SEPARATOR.join([mesh, "edges"])), vector_style="line",
                edge_color=np.array([MESH_EDGE_RGB[int(t)] for t in tags]) / 255, opacity=1.0,
                metadata={MESH_EDGES: True}, **shown),
    ]


@thread_worker
def placed_map(path: Path, level_shapes: list[tuple[int, int]], level0_um: float) -> tuple:
    """The necrosis map on the H&E pyramid level it is shown on, off the UI thread."""
    return call_level(path, level_shapes, level0_um)


class SlideList(QWidget):
    """Slide picker docked into the napari window."""

    def __init__(self, viewer, directory: Path | None = None) -> None:
        """Build the slide picker for the slides indexed under `directory`, or every indexed slide when it is None."""
        super().__init__()
        self.viewer = viewer
        self.directory = str(directory.resolve()) if directory else None
        self._stain = None  # read on first use, then kept
        self._readout = None  # the readout run's numbers, read on first use, then kept
        self._necrosis_areas = None  # the necrosis run's areas per animal, read on first use, then kept
        self._map_worker = None  # the necrosis map being placed on the H&E pyramid
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
        self.status.setWordWrap(True)  # long messages wrap within the dock width
        # the busy line and bar share one row whose room stays reserved when idle, so nothing above moves
        self.activity = QLabel()
        self.activity.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)  # cut at the dock width
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(120)
        for part in (self.activity, self.progress):
            policy = part.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            part.setSizePolicy(policy)
        self._show_activity(None)
        self._buttons = (load, add, clear)
        self._worker = None  # the block being opened, one at a time

        buttons = QHBoxLayout()
        for button in (load, add, clear):
            buttons.addWidget(button)

        # the 2D camera keeps its orientation, so every layer turns about the world origin and the camera centre follows
        self._angle = 0
        self.angle = QSlider(Qt.Orientation.Horizontal)
        self.angle.setRange(-180, 180)
        self.angle.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.angle.setTickInterval(TURN_STEP)
        self.angle.setToolTip("Turn the view clockwise by this angle; the pixels stay as stored")
        self.angle.valueChanged.connect(self._turn_to)
        self.angle_shown = QLabel("0\u00b0")
        self.angle_shown.setMinimumWidth(40)
        left, right = QPushButton("\u27f2 90"), QPushButton("\u27f3 90")
        left.setToolTip("Turn the view 90 degrees counterclockwise")
        right.setToolTip("Turn the view 90 degrees clockwise")
        left.clicked.connect(lambda: self._turn_by(-TURN_STEP))
        right.clicked.connect(lambda: self._turn_by(TURN_STEP))
        turn = QHBoxLayout()
        turn.addWidget(QLabel("Turn"))
        for control in (left, self.angle, right, self.angle_shown):
            turn.addWidget(control)
        self.viewer.layers.events.inserted.connect(self._place_new)
        self.viewer.camera.events.zoom.connect(self._fit_mesh_edges)

        layout = QVBoxLayout(self)  # passing self installs it as this widget's layout
        layout.addLayout(filters)
        layout.addWidget(self.list)
        layout.addWidget(self.info)
        layout.addWidget(self.hide_background)
        layout.addWidget(self.even_staining)
        layout.addLayout(turn)
        layout.addLayout(buttons)
        busy = QHBoxLayout()
        busy.addWidget(self.activity, stretch=1)
        busy.addWidget(self.progress)
        layout.addLayout(busy)
        layout.addWidget(self.status)

        self._pixels = QTimer(self)
        self._pixels.timeout.connect(self._poll_pixels)
        self._pixels.start(PIXEL_POLL_MS)

        self._fill_boxes()
        self.refresh()

    def _fill_boxes(self) -> None:
        """Offer the values the index holds, scoped to the listed directory."""
        scope, params = self._scope()
        where = f"WHERE {scope[0]}" if scope else ""
        for column, box in self.boxes.items():
            # scoped to the widget's directory, as the list is
            values = query(
                f"SELECT DISTINCT {_column(column)} FROM slides {where} ORDER BY 1",
                tuple(params),
            )
            box.blockSignals(True)  # silent during the refill; refresh() runs once after
            box.clear()
            box.addItem(ANY)
            box.addItems([str(row[0]) for row in values if row[0] is not None])
            box.setCurrentIndex(0)
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
                # cast to the column's type, as the box holds text
                params.append(FILTER_TYPES[column](box.currentText()))
        return ("WHERE " + " AND ".join(clauses) if clauses else "", tuple(params))

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
        """A slide's full sidecar with its nested fields such as the registration, or None when absent."""
        path = slide_path(row)
        sidecar = find_sidecar(path.parent, path.name.split(".")[0])
        return Slide(**json.loads(sidecar.read_text())) if sidecar else None

    @classmethod
    def _registration(cls, row) -> Registration | None:
        """The transform from a slide's sidecar."""
        slide = cls._sidecar(row)
        return slide.registration if slide is not None else None

    @staticmethod
    def _label(row) -> str:
        """One list entry per block: species, substance, dose and animal, blank where a slide carries none."""
        dose = row["dose_mg_per_kg"]
        dose_text = f"{dose:>3} mg/kg" if dose is not None else ""
        return f"{row['species']:<6} {row['substance'] or ''} {dose_text}  {row['animal_id']}"

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
                lines.append(f"<br><b>{stain_name(row['stain'])}</b>")
                lines.extend(details)

        animal = animal_name(block, slides[0]["substance"])
        mapped = self._necrosis_record().get(animal)
        if mapped is not None:
            lines.append(f"<br><b>Necrosis</b> (necrosis map, {mapped['written'][:10]})")
            drawn = f", drawn {mapped['area_drawn']:.1%}" if mapped.get("area_drawn") is not None else ""
            lines.append(f"{mapped['area_E']:.1%} of tissue without lumina{drawn}")
        lines.extend(self._readout_lines(slides))
        self.info.setText("<br>".join(lines))

    def _readout_record(self) -> dict:
        """The newest CYP2E1 readout run's record from the prediction directory, read once and kept."""
        if self._readout is None:
            self._readout = {}
            directory = settings.predictions_dir()
            paths = sorted(directory.glob(READOUT_RECORDS)) if directory is not None and directory.is_dir() else []
            for path in paths:
                try:
                    record = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    log.warning("unreadable readout record %s: %s", path, exc)
                    continue
                if record.get("written", "") > self._readout.get("written", ""):
                    self._readout = record
        return self._readout

    def _necrosis_record(self) -> dict:
        """The necrosis run's areas per animal, each with the run's date, read once and kept."""
        if self._necrosis_areas is None:
            self._necrosis_areas = {}
            directory = settings.predictions_dir()
            path = directory / NECROSIS_RECORD if directory is not None else None
            if path is not None and path.exists():
                try:
                    record = json.loads(path.read_text())
                    self._necrosis_areas = {row["animal"]: row | {"written": record["written"]}
                                            for row in record["slides"]}
                except (OSError, ValueError, KeyError) as exc:
                    log.warning("unreadable necrosis record %s: %s", path, exc)
        return self._necrosis_areas

    def _readout_lines(self, slides: list) -> list[str]:
        """Info lines with the CYP2E1-positive share of each DAB slide in the block."""
        record = self._readout_record()
        measured = record.get("slides_measured", {})
        lines = []
        for row in slides:
            found = measured.get(slide_stem(row))
            if row["stain"] not in DAB_STAINS or found is None:
                continue
            stain = stain_name(row["stain"])
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
        slide_key = slide_stem(row)
        pyramids = {SOURCE_LEVELS: levels}
        # fields are fitted on a file's only scene, on its own pixel grid, so they go in before the warp
        flat = self._flattening(path.parent, slide_key) if info.n_scenes == 1 else None
        if flat is not None:
            pyramids[EVENED_LEVELS] = flatten_levels(levels, flat, info.pixel_size_um)
        # applied always, so one slide has one appearance for everyone who opens it
        pyramids = {key: self._matched(p, slide_key) for key, p in pyramids.items()}
        stain = stain_name(row["stain"])
        # a scene number only means something on the files that hold more than one
        what = f"{stain} stain s{scene}" if info.n_scenes > 1 else f"{stain} stain"
        name = layer_name(animal_name(row["serial_block"], row["substance"]), what)
        if reference_um is None:
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
    def _necrosis_map(animal: str) -> Path | None:
        """The necrosis map for one animal, when the prediction directory holds one."""
        directory = settings.predictions_dir()
        path = directory / f"{animal}{NECROSIS_SUFFIX}" if directory is not None else None
        return path if path is not None and path.exists() else None

    def _load_necrosis(self, animal: str, reference_um: float) -> None:
        """Add the animal's necrosis map, when one exists, placed on the H&E pyramid off the UI thread."""
        path = self._necrosis_map(animal)
        if path is None or not self._reference_shapes:
            return
        name = layer_name(animal, NECROSIS_NAME)
        self._set_busy(True)  # Load, Add, Clear and the list stay locked until the map arrives
        self._show_activity(f"{animal}: placing the necrosis map")
        worker = placed_map(path, list(self._reference_shapes), reference_um)
        worker.returned.connect(lambda placed: add_call_layer(self.viewer, *placed, name=name))
        worker.errored.connect(lambda exc: self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}"))
        worker.finished.connect(self._map_done)  # emitted after an error too
        self._map_worker = worker
        worker.start()

    def _map_done(self) -> None:
        """Free the controls once the necrosis map is in."""
        self._map_worker = None
        self._set_busy(False)

    def _load_readouts(self, slides: list) -> None:
        """Add the stored gradient map of each DAB slide, on its H&E partner's grid."""
        import numpy as np

        directory = settings.predictions_dir()
        if directory is None or not directory.is_dir():
            return
        references = {slide_stem(row)
                      for row in slides if row["stain"] == REFERENCE_STAIN}
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
                name=layer_name(animal_name(row["serial_block"], row["substance"]),
                                stain_name(row["stain"])),
                # a pyramid level's pixel q is centred on level-0 pixel factor * q, where napari puts it with no shift
                scale=(pixel_um, pixel_um),
                units="um",
                colormap=STEP_COLOURS,
                opacity=STEP_OPACITY,
                contrast_limits=(0, STEPS),
                interpolation2d="nearest",  # steps are classes, a blend between two would be a third
                visible=False,
            )

    def _load_meshes(self, animal: str) -> None:
        """Add each mesh stored for the animal, in the H&E frame, with its named edges at a fixed screen width."""
        directory = settings.meshes_dir()
        if directory is None or not directory.is_dir():
            return
        for path in sorted(directory.glob(f"{animal}_*{MESH_FIELDS}")):
            try:
                layers = mesh_layers(path, animal)
            except (OSError, ValueError, KeyError) as exc:
                log.exception("could not load mesh %s", path)
                self.status.setText(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            for layer in layers:
                self.viewer.add_layer(layer)
        self._fit_mesh_edges()

    def _fit_mesh_edges(self, event=None) -> None:
        """Set every mesh edge layer's width to MESH_EDGE_PX screen pixels; the camera zoom is screen pixels per um."""
        width = MESH_EDGE_PX / self.viewer.camera.zoom
        for layer in self.viewer.layers:
            if MESH_EDGES in layer.metadata:
                layer.edge_width = width

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
        if not slides or self._worker is not None or self._map_worker is not None:
            return

        animal = animal_name(block, slides[0]["substance"])
        state = {"reference_um": None, "unaligned": [], "done": 0}
        self._reference_shapes = []  # filled from this block's reference as it opens
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
            self._load_meshes(animal)
            if reference_um:
                self._load_necrosis(animal, reference_um)
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
        stain = stain_name(row["stain"])
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
        if self._worker is not None or self._map_worker is not None:
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

    def _turn_by(self, step: int) -> None:
        """Turn the view by a number of degrees clockwise, through the slider so both stay in step."""
        self.angle.setValue((self.angle.value() + step + 180) % 360 - 180)

    def _turn_to(self, angle: int) -> None:
        """Turn every layer to `angle` and the camera with them, so the spot in the middle stays there."""
        import numpy as np

        step = turned(angle - self._angle)
        self._angle = angle
        self.angle_shown.setText(f"{angle}\u00b0")
        for layer in self.viewer.layers:
            if BASE_AFFINE in layer.metadata:  # layers from before this widget keep their place
                self._place(layer)
        centre = list(self.viewer.camera.center)
        centre[-2:] = (step @ np.array([*centre[-2:], 1.0]))[:2]
        self.viewer.camera.center = tuple(centre)

    def _place(self, layer) -> None:
        """Lay the view's turn over a layer's own placement."""
        layer.affine = turned(self._angle) @ layer.metadata[BASE_AFFINE]

    def _place_new(self, event) -> None:
        """Keep a new layer's own placement, then turn it with the view."""
        layer = event.value
        if layer.ndim != 2:  # the turn is 2D; anything else is left as it comes
            return
        layer.metadata[BASE_AFFINE] = layer.affine.affine_matrix.copy()
        if self._angle:
            self._place(layer)

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
