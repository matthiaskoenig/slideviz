"""What a slide sidecar contains, as one declaration."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator

# unknown fields are a validation error; enum fields hold the plain string value
STRICT = ConfigDict(extra="forbid", use_enum_values=True)

# a name the pipeline reads back; empty strings are rejected
Name = Annotated[str, Field(min_length=1)]

# parsed on input, serialized as ISO text
Date = Annotated[date, PlainSerializer(lambda d: d.isoformat(), return_type=str)]
Timestamp = Annotated[datetime, PlainSerializer(lambda d: d.isoformat(), return_type=str)]


class Species(StrEnum):
    """Common name, matching the filename prefix."""

    MOUSE = "mouse"
    RAT = "rat"
    PIG = "pig"
    HUMAN = "human"


class Stain(StrEnum):
    """The stain on the section."""

    HE = "he"
    CYP2E1 = "cyp2e1"  # mouse pericentral zonation marker
    CYP1A2 = "cyp1a2"  # the rat analogue of CYP2E1
    HMGB1 = "hmgb1"  # nuclear-integrity loss, reads out necrosis


class Modality(StrEnum):
    """How the slide was imaged."""

    BRIGHTFIELD = "brightfield"
    FLUORESCENCE = "fluorescence"


class Scene(BaseModel):
    """One independently scanned region. Fields left out fall back to the slide's."""

    model_config = ConfigDict(extra="allow")  # per-scene overrides of any slide field

    scene: int | None = None  # None means take the position in the list


class NonRigid(BaseModel):
    """A displacement field applied after the matrix, stored as its own file."""

    model_config = STRICT

    path: Name  # relative to the sidecar's directory
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # detects a swapped or damaged field file
    # grid on the reference slide from its origin; dx, dy in um, sampled at reference
    # point p, give where the matrix-transformed moving slide is read: p + (dx, dy)
    grid_um_per_px: float = Field(gt=0.0)
    grid_shape_rc: list[int] = Field(min_length=2, max_length=2)
    method: Name  # software, settings and input image, an evaluation criterion
    median_um: float | None = Field(default=None, ge=0.0)
    p95_um: float | None = Field(default=None, ge=0.0)
    registered: Date | None = None
    code: Name | None = None  # git commit of the code that computed the field


class Registration(BaseModel):
    """How this slide maps onto another one. The reference slide carries none."""

    model_config = STRICT

    reference: Name  # serial_block partner this was aligned to, the parent artifact
    matrix: list[list[float]]  # 3x3, x/y at full resolution, onto the reference's grid
    slide_shape_rc: list[int] = Field(min_length=2, max_length=2)  # the resolution the matrix is in
    method: Name  # software and settings, an evaluation criterion
    error_um: float | None = Field(default=None, ge=0.0)  # residual, in physical units
    residual_px: float | None = Field(default=None, ge=0.0)  # how exactly the affine fit the warp
    # tissue-outline overlap after the transform, 0 to 1, for methods reporting no um
    outline_dice: float | None = Field(default=None, ge=0.0, le=1.0)
    registered: Date | None = None
    code: Name | None = None  # git commit of the code that computed the matrix
    nonrigid: NonRigid | None = None  # absent means the matrix is the whole transform

    @field_validator("matrix")
    @classmethod
    def _affine(cls, matrix: list[list[float]]) -> list[list[float]]:
        """3x3, bottom row [0,0,1], and invertible."""
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError(f"must be 3x3, got {len(matrix)}x{[len(r) for r in matrix]}")
        if matrix[2] != [0.0, 0.0, 1.0]:
            raise ValueError(f"bottom row must be [0, 0, 1], got {matrix[2]}")
        if abs(float(np.linalg.det(np.array(matrix, float)))) < 1e-12:
            raise ValueError("singular, so it maps the slide onto a line")
        return matrix

    @field_validator("slide_shape_rc")
    @classmethod
    def _positive(cls, shape: list[int]) -> list[int]:
        """The shape, checked for positive pixel dimensions."""
        if any(v <= 0 for v in shape):
            raise ValueError(f"pixel dimensions must be positive, got {shape}")
        return shape


class Slide(BaseModel):
    """One sidecar. Multi-scene files carry a scenes list and index one row per entry."""

    model_config = STRICT

    file: Name
    species: Species
    substance: Name | None = None  # the treatment, e.g. apap; rat slides carry none
    dose_mg_per_kg: int | None = Field(default=None, ge=0)  # rat slides carry none
    animal_id: Name
    stain: Stain
    modality: Modality
    serial_block: Name
    original_name: str | None = None
    excluded: str | None = None  # why the slide is excluded, e.g. an antibody negative control
    scenes: list[Scene] | None = None
    registration: Registration | None = None

    # Wet-lab identity; an animal can have multiple paraffin-block cases.
    # why animal_id differs from the scanner's name, e.g. a lab labelling swap
    animal_id_corrected: str | None = None
    case_id: str | None = None
    lobes: str | None = None
    antibody_dilution: str | None = None
    staining_run: str | None = None

    # acquisition provenance, read from the vendor file
    acquired: Timestamp | None = None
    scanner_software: str | None = None
    scanner_version: str | None = None
    objective: str | None = None
    magnification: int | None = Field(default=None, gt=0)
    numerical_aperture: float | None = Field(default=None, gt=0.0)
    camera: str | None = None
    pixel_size_um: float | None = Field(default=None, gt=0.0)
    source_compression: str | None = None
    source_path: str | None = None

    # filled in while indexing
    directory: str | None = None
    scene: int = Field(default=0, ge=0)


# SQLite type per python type
SQL_TYPES = {int: "INTEGER", float: "REAL", str: "TEXT"}

# Slide fields with no table column: the nested models and excluded
NESTED = ["scenes", "registration", "excluded"]

# Columns the index holds: the model's fields, minus the nested ones
COLUMNS = [name for name in Slide.model_fields if name not in NESTED]

# What a sidecar must carry. directory and scene are added during indexing.
REQUIRED = [
    name
    for name, field in Slide.model_fields.items()
    if field.is_required() and name not in NESTED
]

# Filled in by the indexer, NOT NULL in the table
DERIVED = ["directory", "scene"]


def _sql_type(name: str) -> str:
    """INTEGER, REAL or TEXT for one field, from the model's annotation."""
    annotation = Slide.model_fields[name].annotation
    for python_type, sql in SQL_TYPES.items():
        if annotation is python_type or python_type in getattr(annotation, "__args__", ()):
            return sql
    return "TEXT"  # enums, dates and anything else are stored as their string value


def create_table_sql() -> str:
    """The CREATE TABLE statement for the current Slide model."""
    lines = []
    for name in COLUMNS:
        not_null = " NOT NULL" if name in REQUIRED or name in DERIVED else ""
        lines.append(f"    {name:<18} {_sql_type(name)}{not_null},")
    return (
        "CREATE TABLE IF NOT EXISTS slides (\n"
        + "\n".join(lines)
        # same filename under two roots is two slides, and a scene is one tissue piece
        + "\n    PRIMARY KEY (directory, file, scene)\n);"
    )
