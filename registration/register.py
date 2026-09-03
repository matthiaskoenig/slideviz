"""Register one serial block. The entry point.

Converts Zarr to OME-TIFF (VALIS cannot read the bf2raw layout), runs
registration/valis_register.py as a subprocess, checks the error against
ERROR_LIMIT_UM, writes the transform into the sidecars, and reports where the
TIFFs can be archived. Above the limit nothing is written.

Stops after one block, so a bad registration is seen before the next starts.

Runs in the slideviz venv, not this folder's. Call it from the repo root, since
`uv run` inside registration/ picks the VALIS venv, which has no slideviz and
fails at the sidecar step after the whole run has finished.

    uv run python registration/register.py 375mg_m1
    uv run python registration/register.py 500mg_m4 --retry    # lock scale, check flips
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
# VALIS pins numpy<2 and slideviz needs numpy>=2, so registration/ is a separate
# uv project with its own venv. They meet through transforms.json on disk.
VALIS_PROJECT = Path(__file__).resolve().parent
TMPDIR = Path("/home/michelle/tmp")

STAINS = ("he", "cyp2e1")  # he is the reference: the morphology and segmentation frame

# a good pair lands near 25 µm and a failed one near 875, so anything in between is
# a bad registration rather than a merely imprecise one
ERROR_LIMIT_UM = 500.0

# A wrong fit can still look good by error alone; the scale on failed blocks was
# 0.813, 1.353 and 1.586, far from the block's true 0.96–1.01.
SCALE_TOLERANCE = 0.10

# a similarity transform is isotropic, so any stretch is a sign the fit went wrong
SHEAR_TOLERANCE = 1.05


def check_geometry(matrix, allow_reflection: bool = False) -> str | None:
    """Why this transform is implausible for two sections off one block, or None.

    Guards what the error value cannot: a wrong fit can be self-consistent and
    report a low error while mapping the slide to the wrong size or handedness.
    """
    import numpy as np

    linear = np.asarray(matrix, float)[:2, :2]
    determinant = float(np.linalg.det(linear))

    if abs(determinant) < 1e-12:
        return "singular, so it maps the slide onto a line"

    # a mirrored section cannot be undone by rotation, so it is a mounting error
    # rather than a registration this script should record
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
    # check=False, since the exit code is reported here rather than as a traceback
    result = subprocess.run(command, check=False, **kwargs)
    if result.returncode != 0:
        sys.exit(f"failed with exit {result.returncode}")


def to_tiff(block: str, stain: str) -> Path:
    """The OME-TIFF for one slide, converted from its Zarr if it is not there yet."""
    name = f"mouse_apap_{block}_{stain}"
    tiff = TIFF_DIR / f"{name}.ome.tiff"
    if tiff.exists():
        print(f"{tiff.name} already present, keeping it")
        return tiff

    zarr = ZARR_DIR / f"{name}.zarr"
    if not zarr.exists():
        sys.exit(f"no zarr for {name}")

    start = time.time()
    # LZW because JPEG_2000 is silently broken in raw2ometiff 0.10.0 and writes
    # uncompressed output
    run(["raw2ometiff", str(zarr), str(tiff), "--compression=LZW", "--max_workers=2"])
    size = tiff.stat().st_size / 1e9
    print(f"{tiff.name}: {size:.1f} GB in {time.time() - start:.0f} s")
    return tiff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", help="serial block, e.g. 375mg_m1")
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
    args = parser.parse_args()

    # the sidecar write at the end needs slideviz; check now rather than after
    # VALIS has run for an hour
    try:
        import slideviz  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "no slideviz in this environment. Run from the repo root:\n"
            "    uv run python registration/register.py <block>"
        )

    TIFF_DIR.mkdir(exist_ok=True)
    TMPDIR.mkdir(exist_ok=True)

    print(f"=== {args.block}: to OME-TIFF ===")
    tiffs = [to_tiff(args.block, stain) for stain in STAINS]

    # VALIS takes a directory, so give it one holding just this block
    slides = RUN_DIR / f"pair_{args.block}" / "slides"
    slides.mkdir(parents=True, exist_ok=True)
    for tiff in tiffs:
        link = slides / tiff.name
        # exists() follows the link, so a link left by an earlier run pointing at a
        # path that has since moved reads as absent and then fails to be created
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(tiff)

    out = RUN_DIR / f"out_{args.block}{args.out_suffix}"  # sibling of the input, never its parent
    out.mkdir(exist_ok=True)

    # Failed pairs can fit a clean but wrong similarity transform. Locking scale
    # keeps the block at the correct size, which is ~1.0 for paired sections.
    extra = ["--fixed-scale", "--check-reflections"] if args.retry else []
    # rematch dies in an SVD that does not converge on these pairs, so any run that
    # is already a retry uses one matcher and skips that pass
    if args.single_matcher or args.detector or args.smooth:
        extra += ["--single-matcher"]
    if args.detector:
        extra += ["--detector", args.detector]
    if args.smooth:
        extra += ["--smooth", str(args.smooth)]
    if args.max_dim:
        extra += ["--max-dim", str(args.max_dim)]

    print(f"\n=== {args.block}: registering ===")
    run(
        ["uv", "run", "python", "valis_register.py", str(slides), str(out),
         "--reference", f"mouse_apap_{args.block}_he.ome.tiff",
         "--rigid-only", "--gradient", *extra],
        cwd=VALIS_PROJECT,
        env={**__import__("os").environ, "TMPDIR": str(TMPDIR)},
    )

    error_um = None
    summary = out / "registration_error.csv"
    if summary.exists():
        import csv

        rows = [r for r in csv.DictReader(summary.open()) if r.get("rigid_D")]
        if rows:
            error_um = float(rows[0]["rigid_D"])

    # a poor fit is worse than none; do not write a wrong transform
    reasons = []
    if error_um is None:
        # no error table means the run crashed, so do not write the transform
        reasons.append("no error reported, so the run did not finish")
    elif error_um > ERROR_LIMIT_UM:
        reasons.append(f"rigid error {error_um:.0f} µm is above the {ERROR_LIMIT_UM:.0f} µm limit")

    print(f"\n=== {args.block}: reading the transform ===")
    from slideviz.data.registration import from_valis_run, write_to_sidecars

    registrations = {}
    if not (out / "transforms.json").exists():
        # VALIS writes this last, so its absence means the run died partway
        reasons.append("no transforms.json, so registration did not complete")
    else:
        registrations = from_valis_run(out, error_um=error_um)
        for name, registration in registrations.items():
            implausible = check_geometry(registration.matrix, allow_reflection=args.retry)
            if implausible:
                reasons.append(f"{name}: {implausible}")

    if reasons:
        print(f"\n=== {args.block}: REGISTRATION FAILED ===")
        for reason in reasons:
            print(f"  {reason}")
        print("  nothing written to the sidecars; the transform would be wrong")
        print(f"  look at {out}/slides/overlaps/slides_rigid_overlap.png")
        sys.exit(1)

    print(f"\n=== {args.block}: writing the transform to the sidecars ===")
    for directory in (ZARR_DIR, CZI_DIR):
        write_to_sidecars(registrations, directory, write=True)

    print(f"\n=== {args.block}: done ===")
    print(f"  rigid error: {error_um:.1f} µm" if error_um else "  no error reported")
    print(f"  overlap:     {out}/slides/overlaps/slides_rigid_overlap.png")
    print(f"  view:        uv run python scripts/view_registered.py {ZARR_DIR} "
          f"mouse_apap_{args.block}_cyp2e1")
    if not args.keep_tiff:
        gb = sum(t.stat().st_size for t in tiffs) / 1e9
        print(f"\n  the TIFFs ({gb:.1f} GB) are only needed by VALIS and can go to the NAS:")
        print(f"    {TIFF_DIR}/mouse_apap_{args.block}_*.ome.tiff")


if __name__ == "__main__":
    main()
