"""Run a set of slides named <dose>mg_<animal>_<lobe>_<stain>.<format> through the pipeline; step 1: the slide list.

    uv run slideviz-run <folder> --species mouse --substance apap --out <run folder>

Step 1 reads every file name and header under <folder> (subfolders too), prints one row per animal and lobe with the
stains found, and writes into <out> only: slides/ (a link and a sidecar per readable slide, named by stem), slides.tsv
(one row per file with its status) and slides.json (the run record). A name that does not fit the pattern stops the
run before anything is written; files in other formats are listed as skipped, slides with a broken header as unreadable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from slideviz import log
from slideviz.data.provenance import code_version
from slideviz.data.schema import Modality, Slide, Species, Stain
from slideviz.io.reader import BACKENDS, slide_info, suffix_of

logger = logging.getLogger(__name__)

PATTERN = "<dose>mg_<animal>_<lobe>_<stain>.<format>"
EXAMPLE = "000mg_m01_left_he.ndpi"
# the lowercased name without its format suffix; animal and lobe are letters, digits and hyphens
NAME = re.compile(r"(?P<dose>\d+)mg_(?P<animal>[a-z0-9-]+)_(?P<lobe>[a-z0-9-]+)_(?P<stain>[a-z0-9]+)")
TOKEN = re.compile(r"[a-z0-9-]+")
STAINS = [s.value for s in Stain]
# names left out of the walk: hidden, Synology metadata (@eaDir) and recycle bins (#recycle)
SKIP = (".", "@", "#")
# scanner pixel sizes in um; a header outside this range holds no physical scale (e.g. 72 dpi)
PIXEL_UM = (0.05, 5.0)
TABLE = ["original_name", "stem", "dose_mg_per_kg", "animal", "lobe", "stain", "pixel_size_um", "width", "height",
         "scenes", "status", "source_path"]


def find_files(folder: Path) -> tuple[list[Path], list[dict]]:
    """Every file under folder, linked folders followed once and a .zarr store counted as one file; and a row per
    folder that could not be listed."""
    found: list[Path] = []
    failed: list[dict] = []
    seen: set[Path] = set()

    def failure(err: OSError) -> None:
        """Record a folder the walk could not list."""
        path = Path(err.filename)
        failed.append({"path": path, "original_name": path.name, "suffix": "",
                       "status": f"unreadable folder: {err.strerror}"})

    for root, dirs, files in os.walk(folder, onerror=failure, followlinks=True):
        here = Path(root)
        if here.resolve() in seen:
            dirs[:] = []
            continue
        seen.add(here.resolve())
        stores = [d for d in dirs if suffix_of(Path(d)) in BACKENDS and not d.startswith(SKIP)]
        found += [here / d for d in stores]
        dirs[:] = [d for d in dirs if d not in stores and not d.startswith(SKIP)]
        found += [here / f for f in files if not f.startswith(SKIP)]
    return sorted(found), failed


def parse(path: Path) -> dict:
    """The name's dose, animal, lobe and stain; a problem when it does not fit; a skip status for other formats."""
    suffix = suffix_of(path)
    row = {"path": path, "original_name": path.name, "suffix": suffix}
    if suffix not in BACKENDS:
        return row | {"status": f"skipped: no reader for '{suffix or path.name}'"}
    base = path.name[: len(path.name) - len(suffix)].lower().removesuffix(".ome")
    match = NAME.fullmatch(base)
    if match is None:
        return row | {"problem": f"does not fit {PATTERN}"}
    if match["stain"] not in STAINS:
        return row | {"problem": f"unknown stain '{match['stain']}' (known: {', '.join(STAINS)})"}
    return row | {"dose_mg_per_kg": int(match["dose"]), "animal": match["animal"], "lobe": match["lobe"],
                  "stain": match["stain"]}


def mark_clashes(rows: list[dict], folder: Path) -> None:
    """Give a problem to a slide named with the same dose, animal, lobe and stain as an earlier one."""
    first: dict[tuple, str] = {}
    for r in rows:
        if "stain" not in r:
            continue
        key = (r["dose_mg_per_kg"], r["animal"], r["lobe"], r["stain"])
        if key in first:
            r["problem"] = f"same dose, animal, lobe and stain as {first[key]}"
        first.setdefault(key, str(r["path"].relative_to(folder)))


def read_header(row: dict) -> dict:
    """The slide's pixel size, level-0 size and scene count, or the status of a header that cannot be used."""
    try:
        info = slide_info(row["path"])
    except Exception as err:  # noqa: BLE001, a damaged or foreign file raises its reader library's own error
        return {"status": f"unreadable: {type(err).__name__}: {err}"}
    size = round(info.pixel_size_um, 6)
    if not PIXEL_UM[0] <= size <= PIXEL_UM[1]:
        return {"status": f"unusable: pixel size {info.pixel_size_um:g} um in the header, outside "
                          f"{PIXEL_UM[0]} to {PIXEL_UM[1]} um"}
    return {"pixel_size_um": size, "width": info.width, "height": info.height, "scenes": info.n_scenes}


def sidecar(r: dict, args: argparse.Namespace) -> dict:
    """The slide's sidecar, checked against the Slide model."""
    car = {"file": r["stem"] + r["suffix"], "species": args.species, "substance": args.substance,
           "dose_mg_per_kg": r["dose_mg_per_kg"], "animal_id": r["animal"], "stain": r["stain"],
           "modality": args.modality, "serial_block": r["block"], "original_name": r["original_name"],
           "lobes": r["lobe"], "pixel_size_um": r["pixel_size_um"], "source_path": r["source_path"]}
    Slide.model_validate(car)
    return car


def planned(rows: list[dict]) -> dict[str, tuple[str, str]]:
    """Every entry slides/ should hold: a link name to its target, a sidecar name to its text."""
    want = {}
    for r in rows:
        if r.get("status", "").startswith("ok"):
            want[r["stem"] + r["suffix"]] = ("link", r["source_path"])
            want[r["stem"] + ".json"] = ("text", json.dumps(r["sidecar"], indent=1) + "\n")
    return want


def conflicts(want: dict[str, tuple[str, str]], folder: Path) -> list[str]:
    """Entries already in folder that are not in the plan or differ from it, each with which of the two."""
    if not folder.exists():
        return []
    bad = []
    for p in sorted(folder.iterdir()):
        kind, value = want.get(p.name, ("", ""))
        if not kind:
            bad.append(f"{p.name} (no slide of this run gives it)")
        elif kind == "link" and not (p.is_symlink() and os.readlink(p) == value):
            bad.append(f"{p.name} (links to another file)")
        elif kind == "text" and not (not p.is_symlink() and p.is_file() and p.read_text() == value):
            bad.append(f"{p.name} (another sidecar text)")
    return bad


def write(want: dict[str, tuple[str, str]], folder: Path) -> None:
    """Create the planned links and sidecars that folder does not hold yet."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, (kind, value) in want.items():
        target = folder / name
        if target.is_symlink() or target.exists():
            continue
        if kind == "link":
            target.symlink_to(value)
        else:
            target.write_text(value)


def overview(rows: list[dict]) -> list[str]:
    """One line per dose, animal and lobe with an x for each stain found."""
    ok = [r for r in rows if r.get("status", "").startswith("ok")]
    stains = [s for s in STAINS if any(r["stain"] == s for r in ok)]
    blocks = sorted({(r["dose_mg_per_kg"], r["animal"], r["lobe"]) for r in ok})
    wa = max([len("animal")] + [len(b[1]) for b in blocks])
    wl = max([len("lobe")] + [len(b[2]) for b in blocks])
    lines = [f"{'dose':>7}  {'animal':<{wa}}  {'lobe':<{wl}}  " + "  ".join(f"{s:>6}" for s in stains)]
    for b in blocks:
        have = {r["stain"] for r in ok if (r["dose_mg_per_kg"], r["animal"], r["lobe"]) == b}
        marks = "  ".join(f"{'x' if s in have else '-':>6}" for s in stains)
        dose = f"{b[0]:03d}mg"
        lines.append(f"{dose:>7}  {b[1]:<{wa}}  {b[2]:<{wl}}  {marks}")
    return lines


def main() -> None:
    """Read the names and headers, print the overview, write the links, sidecars, table and record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help=f"slides named {PATTERN}, e.g. {EXAMPLE}")
    parser.add_argument("--species", required=True, choices=[s.value for s in Species])
    parser.add_argument("--substance", required=True, help="what the dose is of, e.g. apap")
    parser.add_argument("--modality", default=Modality.BRIGHTFIELD.value, choices=[m.value for m in Modality])
    parser.add_argument("--out", type=Path, required=True, help="run folder; nothing is written anywhere else")
    parser.add_argument("--workers", type=int, default=8, help="headers read at once")
    args = parser.parse_args()
    log.setup()

    folder, out = args.folder.resolve(), args.out.resolve()
    if not folder.is_dir():
        sys.exit(f"no folder {folder}")
    if out == folder or folder in out.parents:
        sys.exit(f"--out {out} lies inside {folder}; give a run folder outside it")
    if not TOKEN.fullmatch(args.substance):
        sys.exit(f"--substance '{args.substance}': lowercase letters, digits and hyphens only")

    files, failed = find_files(folder)
    rows = [parse(p) for p in files] + failed
    mark_clashes(rows, folder)
    bad = [r for r in rows if "problem" in r]
    if bad:
        logger.error("%d file names need fixing; nothing was written:", len(bad))
        for r in bad:
            logger.error("  %s: %s", r["path"].relative_to(folder), r["problem"])
        logger.error("Name every slide %s, e.g. %s (controls are 000mg).", PATTERN, EXAMPLE)
        sys.exit(1)
    slides = [r for r in rows if "stain" in r]
    if not slides:
        sys.exit(f"no slides in {folder}; formats read: {', '.join(sorted(BACKENDS))}")

    logger.info("reading %d slide headers in %s", len(slides), folder)
    with ThreadPoolExecutor(args.workers) as pool:
        for r, head in zip(slides, pool.map(read_header, slides)):
            r.update(head)
    for r in slides:
        r["stem"] = (f"{args.species}_{args.substance}_{r['dose_mg_per_kg']:03d}mg_{r['animal']}_{r['lobe']}_"
                     f"{r['stain']}")
        r["block"] = f"{args.substance}_{r['dose_mg_per_kg']:03d}mg_{r['animal']}_{r['lobe']}"
        r["source_path"] = str(r["path"].resolve())
    with_he = {r["block"] for r in slides if r["stain"] == Stain.HE.value and "status" not in r}
    for r in slides:
        if "status" in r:
            continue
        notes = [] if r["block"] in with_he else ["no H&E for this animal and lobe"]
        notes += [f"{r['scenes']} scenes in one file"] if r["scenes"] > 1 else []
        r["status"] = "; ".join(["ok"] + notes)
        r["sidecar"] = sidecar(r, args)

    if not any(r["status"].startswith("ok") for r in slides):
        for r in slides:
            logger.error("  %s: %s", r["path"].relative_to(folder), r["status"])
        sys.exit(f"none of the {len(slides)} slides in {folder} could be read; nothing was written")

    want = planned(rows)
    clash = conflicts(want, out / "slides")
    if clash:
        for r in rows:
            if r["status"] != "ok":
                logger.warning("  %s: %s", r["path"].relative_to(folder), r["status"])
        logger.error("%s differs from this run in %d entries, e.g. %s; nothing was written. Give a new --out.",
                     out / "slides", len(clash), clash[0])
        sys.exit(1)
    write(want, out / "slides")
    with (out / "slides.tsv").open("w", newline="") as handle:
        table = csv.DictWriter(handle, TABLE, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        table.writeheader()
        table.writerows({**r, "original_name": str(r["path"].relative_to(folder))} for r in rows)

    ok = [r for r in rows if r["status"].startswith("ok")]
    counts = {"files": len(rows), "slides": len(ok), "blocks": len({r["block"] for r in ok}),
              "skipped": sum(r["status"].startswith("skipped") for r in rows),
              "unreadable": sum(r["status"].startswith(("unreadable", "unusable")) for r in rows)}
    record = {"step": "slide list", "written": datetime.now(UTC).isoformat(timespec="seconds"),
              "code": code_version(Path(__file__)), "command": sys.argv, "folder": str(folder),
              "species": args.species, "substance": args.substance, "modality": args.modality, **counts}
    (out / "slides.json").write_text(json.dumps(record, indent=1) + "\n")

    for line in overview(rows):
        logger.info(line)
    sizes = [r["pixel_size_um"] for r in ok]
    logger.info("%d slides in %d animal-lobe blocks; pixel size %.4f to %.4f um", counts["slides"], counts["blocks"],
                min(sizes, default=0.0), max(sizes, default=0.0))
    notes = [r for r in rows if r["status"] != "ok"]
    if notes:
        logger.warning("%d files to look at:", len(notes))
        for r in notes:
            logger.warning("  %s: %s", r["path"].relative_to(folder), r["status"])
    logger.info("written: %s/slides/ (links and sidecars), slides.tsv, slides.json", out)


if __name__ == "__main__":
    main()
