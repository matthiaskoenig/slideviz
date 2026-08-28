"""What a slide sidecar contains, as one declaration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Species(StrEnum):
    """Common name, matching the filename prefix rather than the binomial."""

    MOUSE = "mouse"
    RAT = "rat"
    PIG = "pig"
    HUMAN = "human"


class Stain(StrEnum):
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


class Slide(BaseModel):
    """One sidecar. Multi-scene files carry a scenes list and index one row per entry."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    file: str
    species: Species
    substance: str
    dose_mg_per_kg: int
    animal_id: str
    stain: Stain
    modality: Modality
    serial_block: str
    original_name: str | None = None
    scenes: list[Scene] | None = None

    # acquisition provenance, read from the vendor file (the OME conversion does not carry these, so they would otherwise be lost)
    acquired: str | None = None
    scanner_software: str | None = None
    scanner_version: str | None = None
    objective: str | None = None
    magnification: int | None = None
    numerical_aperture: float | None = None
    camera: str | None = None
    source_compression: str | None = None
    source_path: str | None = None

    # filled in while indexing, not written by hand
    directory: str | None = None
    scene: int = 0


# SQLite type per python type, so the DDL follows the model
SQL_TYPES = {int: "INTEGER", float: "REAL", str: "TEXT"}

# Columns the index holds: the model's fields, minus the nested scenes list
COLUMNS = [name for name in Slide.model_fields if name != "scenes"]

# What a sidecar must carry. directory and scene are added during indexing.
REQUIRED = [
    name
    for name, field in Slide.model_fields.items()
    if field.is_required() and name != "scenes"
]

# Filled in by the indexer rather than the sidecar, but never null in the table
DERIVED = ["directory", "scene"]


def _sql_type(name: str) -> str:
    """INTEGER or TEXT for one field, from the model's annotation."""
    annotation = Slide.model_fields[name].annotation
    for python_type, sql in SQL_TYPES.items():
        if annotation is python_type or python_type in getattr(annotation, "__args__", ()):
            return sql
    return "TEXT"  # enums and anything else are stored as their string value


def create_table_sql() -> str:
    """The CREATE TABLE for the current model, so the DDL cannot drift from it."""
    lines = []
    for name in COLUMNS:
        not_null = " NOT NULL" if name in REQUIRED or name in DERIVED else ""
        lines.append(f"    {name:<15} {_sql_type(name)}{not_null},")
    return (
        "CREATE TABLE IF NOT EXISTS slides (\n"
        + "\n".join(lines)
        # same filename under two roots is two slides, and a scene is one tissue piece
        + "\n    PRIMARY KEY (directory, file, scene)\n);"
    )
