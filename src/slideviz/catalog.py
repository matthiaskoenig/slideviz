"""Index JSON sidecars into a queryable SQLite database.

    uv run slideviz-index /path/to/data
    uv run slideviz-index --sql "SELECT file FROM slides WHERE dose_mg_per_kg > 200"

    from slideviz.catalog import build, query
    build(Path("/path/to/data"))
    query("SELECT file FROM slides WHERE dose_mg_per_kg > 200")

The index lives in the user cache directory, or wherever SLIDEVIZ_DB points.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from slideviz.log import setup
from slideviz.schema import COLUMNS, Slide, create_table_sql
from slideviz.settings import settings

log = logging.getLogger(__name__)


def default_db() -> Path:
    """Location of the index, from settings so SLIDEVIZ_DB can move it."""
    return settings.db


SCHEMA = f"""
{create_table_sql()}
CREATE INDEX IF NOT EXISTS idx_block ON slides(serial_block);
CREATE INDEX IF NOT EXISTS idx_dose ON slides(dose_mg_per_kg);
"""


def split_scenes(data: dict) -> list[dict]:
    """One record per scene, or a single scene-0 record when the sidecar lists none."""
    scenes = data.pop("scenes", None)
    if not scenes:
        return [{**data, "scene": 0}]
    # entries override the shared fields, so what is common is written once
    return [{**data, "scene": i, **scene} for i, scene in enumerate(scenes)]


def read_sidecars(directory: Path) -> list[dict]:
    """Load every sidecar under a directory, including nested ones."""
    root = directory.resolve()
    records = []
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text())
        # a slide the lab marked as a control or a failure is kept on disk for its
        # provenance but must never reach an analysis as if it were a readout
        if data.get("excluded"):
            log.info("skipping %s (%s)", path.name, data["excluded"])
            continue
        data["directory"] = str(root)  # the indexed root, so rows stay comparable
        # relative to the root, so a nested layout keeps its subdirectories
        data["file"] = str(path.relative_to(root).with_name(data["file"]))
        records += split_scenes(data)  # a multi-scene sidecar becomes several rows
    return records


def _drop_if_stale(conn: sqlite3.Connection) -> None:
    """Drop the table when its columns no longer match COLUMNS."""
    existing = [r[1] for r in conn.execute("PRAGMA table_info(slides)")]
    if existing and existing != COLUMNS:
        conn.execute("DROP TABLE slides")


def check(records: list[dict]) -> None:
    """Validate every record against the Slide model, naming the file and the field."""
    for record in records:
        name = record.get("file", "<sidecar with no file field>")
        try:
            Slide(**record)
        # without this the insert fails as a bare IntegrityError, naming neither
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ValueError(f"{name}: {problems}") from exc


def build(directory: Path, db_path: Path | None = None) -> int:
    """Rebuild the index from the sidecars in a directory. Returns row count."""
    db_path = db_path or default_db()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    records = read_sidecars(directory)
    check(records)  # fail before touching the database, so the index is not half-baked

    with sqlite3.connect(db_path) as conn:
        _drop_if_stale(conn)  # the index is derived, so an old schema is rebuilt
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
SELECT species, substance, dose_mg_per_kg, stain,
       COUNT(DISTINCT directory || file) AS n,  -- slides, not rows, so scenes do not inflate it
       COUNT(*) AS n_scenes
FROM slides GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
"""


def main() -> None:
    """Rebuild the index from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path, nargs="?", default=settings.data,
                        help="directory of slides and their sidecars (default SLIDEVIZ_DATA)")
    parser.add_argument("--db", type=Path, default=None,
                        help=f"index location (default {default_db()})")
    parser.add_argument("--sql", help="run a query against the index and print the rows")
    parser.add_argument("--check", action="store_true",
                        help="validate the sidecars without building the index")
    parser.add_argument("--schema", action="store_true",
                        help="print the sidecar JSON Schema and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    args = parser.parse_args()
    setup(args.verbose)

    if args.schema:  # generated, so it cannot drift from the model
        print(json.dumps(Slide.model_json_schema(), indent=2))
        return

    if args.sql:
        for row in query(args.sql, db_path=args.db):
            print("  ".join(str(v) for v in row))
        return

    if args.directory is None:
        parser.error("give a directory to index, or --sql to query")
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    if args.check:  # the day a new batch arrives, before it goes near the index
        records = read_sidecars(args.directory)
        check(records)
        log.info("%d records valid", len(records))
        return

    db = args.db or default_db()
    count = build(args.directory, db)
    files = query("SELECT COUNT(DISTINCT directory || file) FROM slides", db_path=db)[0][0]
    scenes = "" if count == files else f" ({count} scenes)"  # they differ only on multi-scene data
    log.info("indexed %d slides%s into %s", files, scenes, db)

    for row in query(SUMMARY_SQL, db_path=db):
        # scene count only where it differs, so single-scene groups read as before
        scenes = "" if row["n_scenes"] == row["n"] else f"  ({row['n_scenes']} scenes)"
        # xxx where the dose is not known yet, matching the placeholder in the filename
        dose = row["dose_mg_per_kg"]
        log.info("  %s %s %3s mg/kg %-7s %d%s", row["species"], row["substance"],
                 dose if dose is not None else "xxx", row["stain"], row["n"], scenes)


if __name__ == "__main__":
    main()
