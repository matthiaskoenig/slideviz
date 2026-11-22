"""Register serial blocks with one method: outline rigid, then DeepFlow non-rigid.

Blocks come from the sidecars in --slides: the slides sharing a serial_block are one
block, its H&E is the reference, and every other slide not marked excluded is
registered onto it. A block with no H&E, several, or nothing else is reported and skipped.
A slide file no sidecar names, or a sidecar without serial_block or stain, is reported,
the rest is registered, and the run exits 1, as it does when nothing was registered.

Works on copies. The run directory gets links to the slides and copies of their
sidecars, and the transforms go into those copies, so the source set stays as it is
until the run has been checked. `<out>/slides/` has the NAS layout (sidecars named
<stem>.json, non-rigid fields in transforms/) and holds finished slides only; a failed
slide's copies move to `<out>/failed/`, and its H&E's too when no slide of the block registered;
copies of a slide marked excluded since an earlier run move to `<out>/excluded/`.

Run from the repo root, so `uv run` picks the slideviz venv; outline_register.py and the
`inputs` step of nonrigid_register.py run in it, the `flow` step in this folder's venv.

    uv run python registration/pipeline.py all --slides /data/michelle/slides \
        --out /data/michelle/registration_run --jobs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from nonrigid_register import GRID_UM, code_version
from outline_register import POSE_EDGE_PX, REFINE_EDGES_PX

from slideviz.data.registration import (
    attach_nonrigid,
    find_sidecar,
    from_valis_run,
    write_to_sidecars,
)
from slideviz.io.reader import BACKENDS, suffix_of

REGISTRATION_PROJECT = Path(__file__).resolve().parent
SLIDEVIZ_PYTHON = [sys.executable]  # the slideviz venv this pipeline runs in
# this folder's venv; --no-sync: main() syncs it once before the blocks run in parallel
REGISTRATION_PYTHON = ["uv", "run", "--no-sync", "python"]
REFERENCE_STAIN = "he"
SCALE_TOLERANCE = 0.10  # true section scales lie between 0.96 and 1.01
SHEAR_TOLERANCE = 1.05  # a similarity transform is isotropic
NONRIGID_INPUT = "haematoxylin"
RIGID_METHOD = (f"outline rigid (slideviz): rotation sweep of the tissue outlines at "
                f"{POSE_EDGE_PX} px, gradient correlation refine at "
                f"{' then '.join(str(e) for e in REFINE_EDGES_PX)} px, scale held at 1")


def check_geometry(matrix) -> str | None:
    """Why this transform is implausible for two sections off one block, or None."""
    linear = np.asarray(matrix, float)[:2, :2]
    determinant = float(np.linalg.det(linear))
    if determinant < 0:
        return f"reflection (det {determinant:+.3f}), so a section may be mounted face down"
    scale = float(np.sqrt(abs(determinant)))
    if abs(scale - 1.0) > SCALE_TOLERANCE:
        return f"scale {scale:.3f} is not within {SCALE_TOLERANCE:.0%} of 1.0"
    larger, smaller = np.linalg.svd(linear, compute_uv=False)
    if smaller > 0 and larger / smaller > SHEAR_TOLERANCE:
        return f"anisotropic by {larger / smaller:.3f}, so the fit is stretched"
    return None


def now() -> str:
    """The current UTC time, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def find_blocks(source: Path) -> tuple[dict[str, dict[str, dict]], dict[str, str],
                                       dict[str, str]]:
    """{serial_block: {stem: sidecar}} in `source`; excluded slides and slides in no block, with why."""
    blocks: dict[str, dict[str, dict]] = {}
    excluded: dict[str, str] = {}
    in_no_block: dict[str, str] = {}
    named = set()
    for path in sorted(source.glob("*.json")):
        name = path.name.removesuffix(".json").removesuffix(".ome")
        if find_sidecar(source, name) != path:
            continue  # <stem>.json wins over <stem>.ome.json, as in find_sidecar
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "file" not in data:
            continue  # predictions and the stain reference sit beside the slides
        named.add(data["file"])
        if data.get("excluded"):
            excluded[name] = str(data["excluded"])
            print(f"{name}: excluded, {data['excluded']}")
        elif not data.get("serial_block") or not data.get("stain"):
            in_no_block[name] = "no serial_block or stain in its sidecar"
        else:
            blocks.setdefault(data["serial_block"], {})[name] = data
    for path in sorted(source.iterdir()):
        if suffix_of(path) in BACKENDS and path.name not in named:
            in_no_block[path.name] = "no sidecar names this slide"
    for name, why in in_no_block.items():
        print(f"{name}: {why}, so in no block")
    return blocks, excluded, in_no_block


def reference_of(sidecars: dict[str, dict]) -> str:
    """The stem of the block's one H&E; ValueError when it has none, several, or nothing else."""
    found = sorted(name for name, data in sidecars.items() if data["stain"] == REFERENCE_STAIN)
    if len(found) != 1:
        raise ValueError(f"{len(found)} H&E slides not excluded ({', '.join(found) or 'none'}), "
                         "so no reference")
    if len(sidecars) == 1:
        raise ValueError(f"only the H&E {found[0]}, nothing to register")
    return found[0]


def uncommitted() -> str:
    """Changed and untracked files in the checkout, as git status lists them."""
    found = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           check=False, cwd=REGISTRATION_PROJECT)
    return found.stdout.strip()


def run_script(python: list[str], command: list[str], log: Path) -> None:
    """Run a script of this folder with `python`, output into `log`; raise on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run([*python, *command], cwd=REGISTRATION_PROJECT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {result.returncode}, see {log}")


def link(path: Path, target: Path) -> None:
    """Point `path` at `target`, replacing an older link and never a file."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"{path} is a file, not a link, so it stays")
    path.symlink_to(target)


def pair_name(name: str, sidecars: dict[str, dict]) -> str:
    """A slide's link name in a pair folder: its stem and its format's suffix."""
    return name + suffix_of(Path(sidecars[name]["file"]))


def prepare(reference: str, slide: str, sidecars: dict[str, dict], source: Path,
            run: Path) -> None:
    """Link the pair's slides into the run and copy their sidecars, registration removed."""
    pair = run / "rigid" / slide / "pair"  # outline_register.py takes a folder of one pair
    for directory in (run / "slides", pair):
        directory.mkdir(parents=True, exist_ok=True)
    for old in pair.iterdir():
        if old.is_symlink():
            old.unlink()  # an earlier run's link, e.g. to a former reference
    (run / "slides" / "transforms" / f"{slide}_nonrigid.npz").unlink(missing_ok=True)
    for name in (reference, slide):
        file = sidecars[name]["file"]
        target = (source / file).resolve()
        if not target.exists():
            raise FileNotFoundError(f"{name}: slide {file} missing in {source}")
        link(run / "slides" / file, target)
        link(pair / pair_name(name, sidecars), target)
        # filled in again by this run only
        data = {key: value for key, value in sidecars[name].items() if key != "registration"}
        (run / "slides" / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


def rigid(reference: str, slide: str, sidecars: dict[str, dict], run: Path, code: str) -> dict:
    """Outline-register one slide onto the reference and write the matrix into its sidecar copy."""
    out = run / "rigid" / slide
    for stale in [out / "transforms.json", out / "outline_error.csv",
                  *out.glob("*_overlap.png")]:
        stale.unlink(missing_ok=True)  # would read as this run's result
    run_script(SLIDEVIZ_PYTHON, ["outline_register.py", str(out / "pair"), str(out),
                                 "--reference", pair_name(reference, sidecars)], out / "log.txt")

    registrations = from_valis_run(out, method=RIGID_METHOD, code=code)
    if list(registrations) != [slide]:
        raise RuntimeError(f"expected {slide} only, got {list(registrations)}")
    registration = registrations[slide]
    implausible = check_geometry(registration.matrix)
    if implausible:
        raise RuntimeError(f"rigid: {implausible}")
    write_to_sidecars(registrations, run / "slides")

    matrix = np.array(registration.matrix)
    with (out / "outline_error.csv").open() as handle:
        correlation = float(next(csv.DictReader(handle))["correlation"])
    return {
        "angle_deg": round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 3),
        "scale": round(float(np.sqrt(abs(np.linalg.det(matrix[:2, :2])))), 4),
        "outline_dice": round(registration.outline_dice, 4),
        "correlation": round(correlation, 4),
    }


def nonrigid(slide: str, run: Path, grid_um: float) -> dict:
    """Compute the DeepFlow field on the new matrix and attach it to the sidecar copy."""
    fields = run / "nonrigid"
    run_script(SLIDEVIZ_PYTHON, ["nonrigid_register.py", "inputs", slide,
                                 "--slides", str(run / "slides"), "--out", str(fields),
                                 "--grid-um", str(grid_um), "--input", NONRIGID_INPUT],
               fields / f"{slide}_inputs_log.txt")
    run_script(REGISTRATION_PYTHON, ["nonrigid_register.py", "flow", slide, "--out", str(fields)],
               fields / f"{slide}_log.txt")
    field = attach_nonrigid(fields / f"{slide}_nonrigid.json", run / "slides")
    return {"median_um": field.median_um, "p95_um": field.p95_um}


def set_aside(names: list[str], run: Path, into: str = "failed") -> None:
    """Move these slides' sidecar copies and fields from slides/ into the run's folder `into`."""
    slides = run / "slides"
    for name in names:
        for path in (slides / f"{name}.json", slides / "transforms" / f"{name}_nonrigid.npz"):
            if path.exists():
                (run / into).mkdir(exist_ok=True)
                shutil.move(path, run / into / path.name)


def register_slide(reference: str, slide: str, sidecars: dict[str, dict], source: Path,
                   run: Path, code: str, grid_um: float) -> dict:
    """Both steps for one slide of a block, as one row of the run record."""
    start = time.time()
    row = {"block": sidecars[slide]["serial_block"], "slide": slide, "reference": reference}
    try:
        prepare(reference, slide, sidecars, source, run)
        row.update(rigid(reference, slide, sidecars, run, code))
        row.update(nonrigid(slide, run, grid_um))
    except Exception as exc:  # noqa: BLE001, any failure is this slide's, recorded in its row
        row["failed"] = f"{type(exc).__name__}: {exc}"
        set_aside([slide], run)
    row.update(code=code, finished=now(), seconds=round(time.time() - start))
    print(f"{slide} {row.get('failed') or 'done'}  ({row['seconds']} s)", flush=True)
    return row


def register_block(sidecars: dict[str, dict], source: Path, run: Path, code: str,
                   grid_um: float) -> list[dict]:
    """Every slide of one block onto its H&E, as rows of the run record."""
    reference = reference_of(sidecars)
    rows = [register_slide(reference, slide, sidecars, source, run, code, grid_um)
            for slide in sorted(sidecars) if slide != reference]
    if all("failed" in row for row in rows):
        set_aside([reference], run)
    return rows


def write_record(run: Path, this_run: dict, rows: list[dict]) -> None:
    """Add this run to pipeline.json, replacing earlier rows of the blocks it registered."""
    path = run / "pipeline.json"
    record = json.loads(path.read_text()) if path.exists() else {"runs": []}
    blocks = {row["block"] for row in rows}
    kept = [row for row in record.get("slides", [])
            if row["block"] not in blocks and row["slide"] not in this_run["excluded"]]
    path.write_text(json.dumps({
        "rigid_method": RIGID_METHOD,
        "nonrigid_input": NONRIGID_INPUT,
        "runs": [*record["runs"], this_run],
        # rows of runs from before the per-slide record, one per block
        **({"blocks": record["blocks"]} if "blocks" in record else {}),
        "slides": sorted([*kept, *rows], key=lambda row: row["slide"]),
    }, indent=1) + "\n")


def main() -> None:
    """Register every named block and add the rows to the run record."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("block", nargs="+",
                        help="a serial_block of the sidecars, e.g. apap_281mg_m2, or 'all'")
    parser.add_argument("--slides", type=Path, required=True,
                        help="slides with their sidecars, named <stem>.json or <stem>.ome.json")
    parser.add_argument("--out", type=Path, required=True, help="run directory")
    parser.add_argument("--grid-um", type=float, default=GRID_UM,
                        help="pixel size of the non-rigid grid in um")
    parser.add_argument("--jobs", type=int, default=1, help="blocks registered at once")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run with uncommitted changes, for tests; the recorded commit "
                             "then differs from the code that ran")
    args = parser.parse_args()

    source, run = args.slides.resolve(), args.out.resolve()
    if not source.is_dir():
        sys.exit(f"no folder {source}")
    if source.is_relative_to(run) or (run / "slides").resolve() == source:
        sys.exit(f"--out {run} contains --slides; the run writes into <out>/slides")
    found, excluded, in_no_block = find_blocks(source)
    if not found:
        sys.exit(f"no blocks found in {source}")
    names = sorted(found) if args.block == ["all"] else args.block
    unknown = [name for name in names if name not in found]
    if unknown:
        sys.exit(f"no sidecar in {source} has serial_block {', '.join(unknown)}")
    blocks, skipped = {}, {}
    for name in names:
        try:
            reference_of(found[name])
        except ValueError as exc:
            skipped[name] = str(exc)
            print(f"{name}: skipped, {exc}")
            continue
        blocks[name] = found[name]
    if not blocks:
        print(f"no block in {source} has one H&E and a slide to register onto it")

    changes = uncommitted()
    if changes and not args.allow_dirty:
        sys.exit(f"uncommitted changes, so the recorded commit would be wrong:\n{changes}")
    code = code_version()

    # one sync and an import check before the blocks start in parallel
    check = subprocess.run(["uv", "run", "python", "-c", "import pyvips, valis"],
                           cwd=REGISTRATION_PROJECT, check=False)
    if check.returncode != 0:
        sys.exit("the registration venv cannot import pyvips and valis")

    print(f"{len(blocks)} blocks, code {code}, into {run}", flush=True)
    this_run = {"started": now(), "host": socket.gethostname(), "command": sys.argv,
                "code": code, "source": str(source), "grid_um": args.grid_um,
                "blocks": list(blocks), "skipped": skipped, "excluded": excluded,
                "in_no_block": in_no_block}
    run.mkdir(parents=True, exist_ok=True)
    set_aside(sorted(excluded), run, "excluded")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = [row for block_rows in pool.map(
            lambda b: register_block(blocks[b], source, run, code, args.grid_um), blocks)
            for row in block_rows]
    this_run["finished"] = now()
    if code_version() != code:
        this_run["code_after"] = code_version()
        print(f"the checkout moved to {this_run['code_after']} during the run", flush=True)
    write_record(run, this_run, rows)

    width = max((len(row["slide"]) for row in rows), default=5)
    print(f"\n{'slide':{width}s} {'angle':>9s} {'scale':>7s} {'dice':>7s} {'median':>8s} "
          f"{'p95':>7s}")
    for row in rows:
        if "failed" in row:
            print(f"{row['slide']:{width}s} FAILED: {row['failed']}")
            continue
        print(f"{row['slide']:{width}s} {row['angle_deg']:9.2f} {row['scale']:7.4f} "
              f"{row['outline_dice']:7.4f} {row['median_um']:6.1f} µm {row['p95_um']:5.1f} µm")
    failed = sum("failed" in row for row in rows)
    print(f"\n{len(rows) - failed} of {len(rows)} slides registered, record in "
          f"{run / 'pipeline.json'}")
    if skipped:
        print(f"skipped, see above: {', '.join(skipped)}")
    if in_no_block:
        print(f"in no block, see above: {', '.join(in_no_block)}")
    if not rows or failed or in_no_block or "code_after" in this_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
