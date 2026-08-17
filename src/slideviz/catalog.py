"""Index JSON sidecars into a queryable SQLite database.

    from slideviz.catalog import build, query
    build(Path("/path/to/data"), Path("slides.db"))
    query(db, "SELECT file FROM slides WHERE dose_mg_per_kg > 200")
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS slides (
    file            TEXT PRIMARY KEY,
    directory       TEXT NOT NULL,
    species         TEXT NOT NULL,
    substance       TEXT NOT NULL,
    dose_mg_per_kg  INTEGER NOT NULL,
    animal_id       TEXT NOT NULL,
    stain           TEXT NOT NULL,
    modality        TEXT NOT NULL,
    serial_block    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_block ON slides(serial_block);
CREATE INDEX IF NOT EXISTS idx_dose ON slides(dose_mg_per_kg);
"""

# Indexed for querying
COLUMNS = [
    "file", "directory", "species", "substance",
    "dose_mg_per_kg", "animal_id", "stain", "modality", "serial_block",
]


def read_sidecars(directory: Path) -> list[dict]:
    """Load every sidecar in a directory."""
    records = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        # Record where the sidecar was found
        data["directory"] = str(directory.resolve())
        records.append(data)
    return records


def build(directory: Path, db_path: Path) -> int:
    """Rebuild the index from the sidecars in a directory. Returns row count."""
    records = read_sidecars(directory)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # Clear this directory's rows, then reinsert from the sidecars
        conn.execute("DELETE FROM slides WHERE directory = ?", (str(directory.resolve()),))
        conn.executemany(
            f"INSERT OR REPLACE INTO slides ({','.join(COLUMNS)}) "
            f"VALUES ({','.join('?' * len(COLUMNS))})",
            [tuple(r.get(c) for c in COLUMNS) for r in records],
        )

    return len(records)


def query(db_path: Path, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Run a query and return the rows, accessible by column name."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


def slide_path(row: sqlite3.Row) -> Path:
    """Full path to the image a row describes."""
    return Path(row["directory"]) / row["file"]
