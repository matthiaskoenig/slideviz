"""Register one serial block. The entry point.

Converts Zarr to OME-TIFF, runs valis_register.py or outline_register.py as a
subprocess, checks the error against ERROR_LIMIT_UM, writes the transform into the
sidecars. Above the limit nothing is written. Stops after one block.

Call it from the repo root, so `uv run` picks the slideviz venv that the sidecar
step needs.

    uv run python registration/register.py 375mg_m1
    uv run python registration/register.py 500mg_m4 --retry    # lock scale, check flips
    uv run python registration/register.py 089mg_m3 --from-outline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

IMAGES = Path("/home/michelle/Projects/image-analysis/images/mouse")
ZARR_DIR = IMAGES / "APAP_zarr"
TIFF_DIR = IMAGES / "APAP_tiff"
CZI_DIR = Path("/home/michelle/Projects/image-analysis/images/mouse/APAP")
RUN_DIR = Path("/home/michelle/Projects/image-analysis/valis_runs")
# a separate uv project: VALIS pins numpy<2, slideviz needs numpy>=2. They meet
# through transforms.json on disk.
VALIS_PROJECT = Path(__file__).resolve().parent
TMPDIR = Path("/home/michelle/tmp")

STAINS = ("he", "cyp2e1")  # he is the reference, the morphology frame
STAIN_NAMES = {"he": "HE", "cyp2e1": "Cyp2e1"}  # as the wet lab writes them

ERROR_LIMIT_UM = 500.0  # a good pair lands near 25 µm, a failed one near 875

# failed blocks came back at 0.813, 1.353 and 1.586, against a true 0.96 to 1.01
SCALE_TOLERANCE = 0.10

SHEAR_TOLERANCE = 1.05  # a similarity transform is isotropic

OUTLINE_METHOD = "outline rigid + gradient NCC refine (slideviz)"


def check_geometry(matrix, allow_reflection: bool = False) -> str | None:
    """Why this transform is implausible for two sections off one block, or None.

    Catches a self-consistent fit that reports a low error while mapping the slide
    to the wrong size or handedness.
    """
    import numpy as np

    linear = np.asarray(matrix, float)[:2, :2]
    determinant = float(np.linalg.det(linear))

    if abs(determinant) < 1e-12:
        return "singular, so it maps the slide onto a line"

    # a mirrored section is a mounting error, beyond what rotation can undo
    if determinant < 0 and not allow_reflection:
        return f"reflection (det {determinant:+.3f}), so a section may be mounted face down"

    scale = float(np.sqrt(abs(determinant)))
    if abs(scale - 1.0) > SCALE_TOLERANCE:
        return f"scale {scale:.3f} is not within {SCALE_TOLERANCE:.0%} of 1.0"

    larger, smaller = np.linalg.svd(linear, compute_uv=False)
    if smaller > 0 and larger / smaller > SHEAR_TOLERANCE:
        return f"anisotropic by {larger / smaller:.3f}, so the fit is stretched"

    return None


def run(command: list[str], **kwargs) -> None:
    """Run a command, showing it first, and stop the script if it fails."""
    print(f"\n$ {' '.join(str(c) for c in command)}\n", flush=True)
    # check=False, so the exit code is reported here
    result = subprocess.run(command, check=False, **kwargs)
    if result.returncode != 0:
        sys.exit(f"failed with exit {result.returncode}")


def attempt(command: list[str], **kwargs) -> int:
    """Run a command, showing it first, and return its exit code."""
    print(f"\n$ {' '.join(str(c) for c in command)}\n", flush=True)
    return subprocess.run(command, check=False, **kwargs).returncode


def readable(path: Path) -> bool:
    """Whether the file opens as a pyramidal slide."""
    import subprocess

    probe = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import pyvips; pyvips.Image.new_from_file({str(path)!r}, page=0).width"],
        cwd=VALIS_PROJECT, capture_output=True, check=False,
    )
    return probe.returncode == 0


def all_blocks() -> set[str]:
    """Every block whose two stains are both present and readable."""
    stems = {p.name.split(".")[0] for p in TIFF_DIR.glob("mouse_apap_*_he.ome.tiff")}
    blocks = {s.removeprefix("mouse_apap_").removesuffix("_he") for s in stems}
    return {
        b for b in blocks
        if all(readable(TIFF_DIR / f"mouse_apap_{b}_{stain}.ome.tiff")
               for stain in STAINS
               if (TIFF_DIR / f"mouse_apap_{b}_{stain}.ome.tiff").exists())
        and (TIFF_DIR / f"mouse_apap_{b}_cyp2e1.ome.tiff").exists()
    }


def to_tiff(block: str, stain: str) -> Path:
    """The OME-TIFF for one slide, converting from its Zarr when missing."""
    name = f"mouse_apap_{block}_{stain}"
    tiff = TIFF_DIR / f"{name}.ome.tiff"
    if tiff.exists():
        print(f"{tiff.name} already present, keeping it")
        return tiff

    zarr = ZARR_DIR / f"{name}.zarr"
    if not zarr.exists():
        sys.exit(f"no zarr for {name}")

    start = time.time()
    # LZW: JPEG_2000 is silently broken in raw2ometiff 0.10.0, writing uncompressed
    run(["raw2ometiff", str(zarr), str(tiff), "--compression=LZW", "--max_workers=2"])
    size = tiff.stat().st_size / 1e9
    print(f"{tiff.name}: {size:.1f} GB in {time.time() - start:.0f} s")
    return tiff


def ensure_sidecar(block: str, stain: str) -> None:
    """Create a missing sidecar for one slide, from its block's H&E sidecar."""
    import json

    tiff = TIFF_DIR / f"mouse_apap_{block}_{stain}.ome.tiff"
    sidecar = TIFF_DIR / f"mouse_apap_{block}_{stain}.ome.json"
    reference = TIFF_DIR / f"mouse_apap_{block}_he.ome.json"
    if sidecar.exists() or not tiff.exists() or not reference.exists():
        return

    data = json.loads(reference.read_text())
    data["file"] = tiff.name
    data["stain"] = stain
    # the wet lab's name differs only in the stain, and the quoting is theirs
    data["original_name"] = data["original_name"].replace(" HE'", f" {STAIN_NAMES[stain]}'")
    sidecar.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  created {sidecar.name}")


def run_one(block: str, out: Path, slides: Path, from_outline: bool,
            extra: list[str], allow_reflection: bool) -> tuple[dict, list[str], float | None]:
    """Run one registration and read back its transform, gates and error."""
    from slideviz.data.registration import VALIS_METHOD, from_valis_run

    script = "outline_register.py" if from_outline else "valis_register.py"
    settings = [] if from_outline else ["--rigid-only", "--gradient", *extra]
    attempt(
        ["uv", "run", "python", script, str(slides), str(out),
         "--reference", f"mouse_apap_{block}_he.ome.tiff", *settings],
        cwd=VALIS_PROJECT,
        env={**__import__("os").environ, "TMPDIR": str(TMPDIR)},
    )

    error_um = None
    summary = out / "registration_error.csv"
    if summary.exists() and not from_outline:
        import csv

        rows = [r for r in csv.DictReader(summary.open()) if r.get("rigid_D")]
        if rows:
            error_um = float(rows[0]["rigid_D"])

    reasons = []
    if error_um is None and not from_outline:
        # a missing error table means the run crashed
        reasons.append("no error reported, so the run did not finish")
    elif error_um is not None and error_um > ERROR_LIMIT_UM:
        reasons.append(f"rigid error {error_um:.0f} µm is above the {ERROR_LIMIT_UM:.0f} µm limit")

    registrations = {}
    if not (out / "transforms.json").exists():
        # VALIS writes this last, so its absence means the run died partway
        reasons.append("no transforms.json, so registration did not complete")
    else:
        method = OUTLINE_METHOD if from_outline else VALIS_METHOD
        registrations = from_valis_run(out, error_um=error_um, method=method)
        for name, registration in registrations.items():
            implausible = check_geometry(registration.matrix, allow_reflection=allow_reflection)
            if implausible:
                reasons.append(f"{name}: {implausible}")

    return registrations, reasons, error_um


def register_block(block: str, args) -> dict:
    """Register one block, falling back to the outline path when VALIS fails.

    Returns a row for the batch report: block, method, error_um, reasons.
    """
    print(f"\n=== {block}: to OME-TIFF ===")
    tiffs = [to_tiff(block, stain) for stain in STAINS]
    for stain in STAINS:
        ensure_sidecar(block, stain)

    # VALIS takes a directory, so give it one holding just this block
    slides = RUN_DIR / f"pair_{block}" / "slides"
    slides.mkdir(parents=True, exist_ok=True)
    for tiff in tiffs:
        link = slides / tiff.name
        # exists() follows the link, so a link left by an earlier run pointing at a
        # path that has since moved reads as absent and then fails to be created
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(tiff)

    extra = ["--fixed-scale", "--check-reflections"] if args.retry else []
    if args.single_matcher or args.detector or args.smooth:
        extra += ["--single-matcher"]
    if args.detector:
        extra += ["--detector", args.detector]
    if args.smooth:
        extra += ["--smooth", str(args.smooth)]
    if args.max_dim:
        extra += ["--max-dim", str(args.max_dim)]

    routes = ["outline"] if args.from_outline else ["valis", "outline"]
    if args.no_fallback:
        routes = routes[:1]

    for route in routes:
        from_outline = route == "outline"
        suffix = args.out_suffix + ("_outline" if from_outline else "")
        out = RUN_DIR / f"out_{block}{suffix}"  # a sibling of the input
        out.mkdir(exist_ok=True)

        # an earlier run's transforms.json would read as this one's result
        for stale in ("transforms.json", "registration_error.csv", "outline_error.csv"):
            (out / stale).unlink(missing_ok=True)

        print(f"\n=== {block}: registering ({route}) ===")
        registrations, reasons, error_um = run_one(
            block, out, slides, from_outline, extra, args.retry,
        )

        if not reasons:
            print(f"\n=== {block}: writing the transform to the sidecars ===")
            from slideviz.data.registration import write_to_sidecars

            # every directory that holds sidecars for this block
            for directory in (ZARR_DIR, TIFF_DIR, CZI_DIR):
                if directory.exists():
                    write_to_sidecars(registrations, directory, write=True)
            print(f"  error: {error_um:.1f} µm" if error_um else "  error: a correlation, see outline_error.csv")
            return {"block": block, "route": route, "error_um": error_um, "reasons": []}

        print(f"\n=== {block}: {route} FAILED ===")
        for reason in reasons:
            print(f"  {reason}")
        if route != routes[-1]:
            print(f"  falling back to the outline path")

    return {"block": block, "route": "none", "error_um": None, "reasons": reasons}


def main() -> None:
    """Convert, register, check and write the transform for one block or many."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", nargs="+", help="serial blocks, e.g. 375mg_m1, or 'all'")
    parser.add_argument("--keep-tiff", action="store_true",
                        help="do not print the archive reminder")
    parser.add_argument("--retry", action="store_true",
                        help="lock scale and check flips, for a pair that failed")
    parser.add_argument("--out-suffix", default="",
                        help="write to out_<block><suffix>, keeping the previous run")
    parser.add_argument("--single-matcher", action="store_true",
                        help="sort and match with one detector, skipping the rematch "
                             "pass that crashes on SVD non-convergence")
    parser.add_argument("--smooth", type=float, default=None,
                        help="blur sigma before the gradient, suppressing fine texture")
    parser.add_argument("--detector", default=None,
                        help="feature detector, e.g. DiskFD; implies --single-matcher")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="longest edge used for matching; raise it when features "
                             "are abundant but not distinctive")
    parser.add_argument("--from-outline", action="store_true",
                        help="align on the tissue outline, skipping VALIS")
    parser.add_argument("--no-fallback", action="store_true",
                        help="stop when VALIS fails, leaving the outline path unrun")
    args = parser.parse_args()

    # the sidecar write needs slideviz; check now, before an hour of VALIS
    try:
        import slideviz  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "no slideviz in this environment. Run from the repo root:\n"
            "    uv run python registration/register.py <block>"
        )

    TIFF_DIR.mkdir(exist_ok=True)
    TMPDIR.mkdir(exist_ok=True)

    blocks = sorted(all_blocks()) if args.block == ["all"] else args.block
    rows = [register_block(block, args) for block in blocks]

    if len(rows) > 1:
        print("\n=== summary ===")
        print(f"{'block':12s} {'route':8s} {'error':>10s}  reasons")
        for row in rows:
            error = f"{row['error_um']:.1f} µm" if row["error_um"] else "-"
            print(f"{row['block']:12s} {row['route']:8s} {error:>10s}  "
                  f"{'; '.join(row['reasons'])}")

    if any(row["route"] == "none" for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
