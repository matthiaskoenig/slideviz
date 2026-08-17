"""Index JSON sidecars into a queryable SQLite database.

    uv run slideviz-index /path/to/data
    uv run slideviz-index --sql "SELECT file FROM slides WHERE dose_mg_per_kg > 200"

    from slideviz.catalog import build, query
    build(Path("/path/to/data"))
    query("SELECT file FROM slides WHERE dose_mg_per_kg > 200")

The index lives in ~/.cache/slideviz/slides.db.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def default_db() -> Path:
    """Location of the index, honouring XDG_CACHE_HOME."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "slideviz" / "slides.db"

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


def build(directory: Path, db_path: Path | None = None) -> int:
    """Rebuild the index from the sidecars in a directory. Returns row count."""
    db_path = db_path or default_db()
    db_path.parent.mkdir(parents=True, exist_ok=True)
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


def query(sql: str, params: tuple = (), db_path: Path | None = None) -> list[sqlite3.Row]:
    """Run a query and return the rows, accessible by column name."""
    with sqlite3.connect(db_path or default_db()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


def slide_path(row: sqlite3.Row) -> Path:
    """Full path to the image a row describes."""
    return Path(row["directory"]) / row["file"]


SUMMARY_SQL = """
SELECT species, substance, dose_mg_per_kg, stain, COUNT(*) AS n
FROM slides GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
"""


def main() -> None:
    """Rebuild the index from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path, nargs="?",
                        help="directory of slides and their sidecars")
    parser.add_argument("--db", type=Path, default=None,
                        help=f"index location (default {default_db()})")
    parser.add_argument("--sql", help="run a query against the index and print the rows")
    args = parser.parse_args()

    if args.sql:
        for row in query(args.sql, db_path=args.db):
            print("  ".join(str(v) for v in row))
        return

    if args.directory is None:
        parser.error("give a directory to index, or --sql to query")
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    db = args.db or default_db()
    count = build(args.directory, db)
    print(f"indexed {count} slides into {db}")

    for row in query(SUMMARY_SQL, db_path=db):
        print(f"  {row['species']} {row['substance']} "
              f"{row['dose_mg_per_kg']:>3} mg/kg {row['stain']:<7} {row['n']}")


if __name__ == "__main__":
    main()
