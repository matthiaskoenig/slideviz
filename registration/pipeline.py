"""Register serial blocks with one method: outline rigid, then DeepFlow non-rigid.

Works on copies. The run directory gets links to the slides and copies of their
sidecars, and the transforms go into those copies, so the source set stays as it is
until the run has been checked. `<out>/slides/` has the NAS layout (sidecars named
<stem>.json, non-rigid fields in transforms/) and holds finished blocks only; a failed
block's copies move to `<out>/failed/`.

Run from the repo root, so `uv run` picks the slideviz venv the sidecar steps need;
outline_register.py and nonrigid_register.py run in this folder's venv.

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
from nonrigid_register import EDGE_PX, code_version
from outline_register import POSE_EDGE_PX, REFINE_EDGES_PX
from register import check_geometry

from slideviz.data.registration import (
    attach_nonrigid,
    from_valis_run,
    write_to_sidecars,
)

REGISTRATION_PROJECT = Path(__file__).resolve().parent
STAINS = ("he", "cyp2e1")  # he is the reference
NONRIGID_INPUT = "haematoxylin"
RIGID_METHOD = (f"outline rigid (slideviz): rotation sweep of the tissue outlines at "
                f"{POSE_EDGE_PX} px, gradient correlation refine at "
                f"{' then '.join(str(e) for e in REFINE_EDGES_PX)} px, scale held at 1")


def now() -> str:
    """The current UTC time, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def stem(block: str, stain: str) -> str:
    """A slide's file stem, e.g. mouse_apap_281mg_m2_he."""
    return f"mouse_apap_{block}_{stain}"


def find_blocks(source: Path) -> list[str]:
    """Every block with a slide of either stain in `source`; prepare() fails unpaired ones."""
    found = set()
    for stain in STAINS:
        found |= {p.name.removeprefix("mouse_apap_").removesuffix(f"_{stain}.ome.tiff")
                  for p in source.glob(f"mouse_apap_*_{stain}.ome.tiff")}
    return sorted(found)


def uncommitted() -> str:
    """Changed and untracked files in the checkout, as git status lists them."""
    found = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           check=False, cwd=REGISTRATION_PROJECT)
    return found.stdout.strip()


def in_registration_venv(command: list[str], log: Path) -> None:
    """Run a script of this folder in its venv, output into `log`; raise on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        # --no-sync: parallel blocks would otherwise sync the venv at the same time
        result = subprocess.run(["uv", "run", "--no-sync", "python", *command],
                                cwd=REGISTRATION_PROJECT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {result.returncode}, see {log}")


def prepare(block: str, source: Path, run: Path) -> None:
    """Link the pair's slides into the run and copy their sidecars, registration removed."""
    pair = run / "rigid" / block / "pair"  # outline_register.py takes a folder of one pair
    for directory in (run / "slides", pair):
        directory.mkdir(parents=True, exist_ok=True)
    (run / "slides" / "transforms" / f"{stem(block, 'cyp2e1')}_nonrigid.npz").unlink(
        missing_ok=True)
    for stain in STAINS:
        name = stem(block, stain)
        tiff = (source / f"{name}.ome.tiff").resolve()
        sidecars = [p for p in (source / f"{name}.json", source / f"{name}.ome.json")
                    if p.exists()]
        if not tiff.exists() or not sidecars:
            raise FileNotFoundError(f"{name}: slide or sidecar missing in {source}")
        for directory in (run / "slides", pair):
            link = directory / f"{name}.ome.tiff"
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise RuntimeError(f"{link} is a file, not a link, so it stays")
            link.symlink_to(tiff)
        data = json.loads(sidecars[0].read_text())
        data.pop("registration", None)  # filled in again by this run only
        (run / "slides" / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


def rigid(block: str, run: Path, code: str) -> dict:
    """Outline-register the pair and write the matrix into the sidecar copy."""
    out = run / "rigid" / block
    for stale in [out / "transforms.json", out / "outline_error.csv",
                  *out.glob("*_overlap.png")]:
        stale.unlink(missing_ok=True)  # would read as this run's result
    in_registration_venv(["outline_register.py", str(out / "pair"), str(out),
                          "--reference", f"{stem(block, 'he')}.ome.tiff"], out / "log.txt")

    registrations = from_valis_run(out, method=RIGID_METHOD, code=code)
    if list(registrations) != [stem(block, "cyp2e1")]:
        raise RuntimeError(f"expected the CYP2E1 slide only, got {list(registrations)}")
    registration = registrations[stem(block, "cyp2e1")]
    implausible = check_geometry(registration.matrix)
    if implausible:
        raise RuntimeError(f"rigid: {implausible}")
    write_to_sidecars(registrations, run / "slides", write=True)

    matrix = np.array(registration.matrix)
    with (out / "outline_error.csv").open() as handle:
        correlation = float(next(csv.DictReader(handle))["correlation"])
    return {
        "angle_deg": round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 3),
        "scale": round(float(np.sqrt(abs(np.linalg.det(matrix[:2, :2])))), 4),
        "outline_dice": round(registration.outline_dice, 4),
        "correlation": round(correlation, 4),
    }


def nonrigid(block: str, run: Path, edge: int) -> dict:
    """Compute the DeepFlow field on the new matrix and attach it to the sidecar copy."""
    fields = run / "nonrigid"
    in_registration_venv(["nonrigid_register.py", block, "--slides", str(run / "slides"),
                          "--out", str(fields), "--edge", str(edge),
                          "--input", NONRIGID_INPUT], fields / f"{block}_log.txt")
    field = attach_nonrigid(fields / f"{stem(block, 'cyp2e1')}_nonrigid.json", run / "slides",
                            write=True)
    return {"median_um": field.median_um, "p95_um": field.p95_um}


def set_aside(block: str, run: Path) -> None:
    """Move a failed block's sidecar copies and field from slides/ into failed/."""
    failed = run / "failed"
    failed.mkdir(exist_ok=True)
    slides = run / "slides"
    for path in [*(slides / f"{stem(block, s)}.json" for s in STAINS),
                 slides / "transforms" / f"{stem(block, 'cyp2e1')}_nonrigid.npz"]:
        if path.exists():
            shutil.move(path, failed / path.name)


def register_block(block: str, source: Path, run: Path, code: str, edge: int) -> dict:
    """Both steps for one block, as one row of the run record."""
    start = time.time()
    row = {"block": block}
    try:
        prepare(block, source, run)
        row.update(rigid(block, run, code))
        row.update(nonrigid(block, run, edge))
    except Exception as exc:  # noqa: BLE001, any failure is this block's, recorded in its row
        row["failed"] = f"{type(exc).__name__}: {exc}"
        set_aside(block, run)
    row.update(code=code, finished=now(), seconds=round(time.time() - start))
    print(f"{block:10s} {row.get('failed') or 'done'}  ({row['seconds']} s)", flush=True)
    return row


def write_record(run: Path, this_run: dict, rows: list[dict]) -> None:
    """Add this run to pipeline.json, replacing earlier rows of the same blocks."""
    path = run / "pipeline.json"
    record = json.loads(path.read_text()) if path.exists() else {"runs": [], "blocks": []}
    blocks = {r["block"]: r for r in record["blocks"]} | {r["block"]: r for r in rows}
    path.write_text(json.dumps({
        "rigid_method": RIGID_METHOD,
        "nonrigid_input": NONRIGID_INPUT,
        "runs": [*record["runs"], this_run],
        "blocks": [blocks[b] for b in sorted(blocks)],
    }, indent=1) + "\n")


def main() -> None:
    """Register every named block and add the rows to the run record."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("block", nargs="+", help="e.g. 281mg_m2, or 'all'")
    parser.add_argument("--slides", type=Path, required=True,
                        help="OME-TIFFs with their sidecars, named <stem>.json or <stem>.ome.json")
    parser.add_argument("--out", type=Path, required=True, help="run directory")
    parser.add_argument("--edge", type=int, default=EDGE_PX,
                        help="longest edge of the non-rigid grid in px")
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
    blocks = find_blocks(source) if args.block == ["all"] else args.block
    if not blocks:
        sys.exit(f"no blocks found in {source}")

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
                "code": code, "source": str(source), "edge_px": args.edge, "blocks": blocks}
    run.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda b: register_block(b, source, run, code, args.edge), blocks))
    this_run["finished"] = now()
    if code_version() != code:
        this_run["code_after"] = code_version()
        print(f"the checkout moved to {this_run['code_after']} during the run", flush=True)
    write_record(run, this_run, rows)

    print(f"\n{'block':10s} {'angle':>9s} {'scale':>7s} {'dice':>7s} {'median':>8s} {'p95':>7s}")
    for row in rows:
        if "failed" in row:
            print(f"{row['block']:10s} FAILED: {row['failed']}")
            continue
        print(f"{row['block']:10s} {row['angle_deg']:9.2f} {row['scale']:7.4f} "
              f"{row['outline_dice']:7.4f} {row['median_um']:6.1f} µm {row['p95_um']:5.1f} µm")
    failed = sum("failed" in row for row in rows)
    print(f"\n{len(rows) - failed} of {len(rows)} registered, record in {run / 'pipeline.json'}")
    if failed or "code_after" in this_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
