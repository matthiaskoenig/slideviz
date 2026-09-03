"""Register serial sections with VALIS and save transforms/QC outputs.

Usage: python valis_register.py <slide_dir> <out_dir> [--reference NAME]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def save_transforms(registrar, path) -> None:
    """Write numeric transforms separately because VALIS pickles need a JVM."""
    import json

    import numpy as np

    slides = {}
    for name, slide in registrar.slide_dict.items():
        width, height = (int(v) for v in slide.slide_dimensions_wh[0])

        # Fit from warp_xy: unlike slide.M, it includes the crop to the reference grid.
        # The full warp is affine, so a grid of points recovers the matrix exactly.
        grid = np.array(
            [[x, y] for x in np.linspace(0, width, 5) for y in np.linspace(0, height, 5)],
            float,
        )
        try:
            warped = np.asarray(
                slide.warp_xy(grid, slide_level=0, pt_level=0, non_rigid=False,
                              crop="reference"),
                float,
            )
        # a rigid stage that failed leaves no crop mask, and there is no transform to save
        except (TypeError, AttributeError, ValueError) as exc:
            print(f"  no transform for {name}: registration did not complete ({exc})")
            return
        padded = np.hstack([grid, np.ones((len(grid), 1))])
        fit, *_ = np.linalg.lstsq(padded, warped, rcond=None)
        matrix = np.vstack([fit.T, [0, 0, 1]])

        slides[name] = {
            "source": str(slide.src_f),
            # x,y at full resolution, mapping this slide onto the reference's grid
            "matrix": matrix.tolist(),
            "residual_px": float(np.abs(padded @ matrix.T - np.hstack(
                [warped, np.ones((len(warped), 1))])).max()),
            "slide_shape_rc": [height, width],
        }

    path.write_text(json.dumps({
        "reference": registrar.reference_img_f,
        # the matrices map each slide onto the reference, in its own pixel grid
        "slides": slides,
    }, indent=2) + "\n")
    print(f"wrote {path.name} ({len(slides)} slides)")


def patch_lightglue_dtype() -> None:
    """Cast reflection-path keypoints to float32 for LightGlue compatibility."""
    import numpy as np
    from valis import feature_matcher

    original = feature_matcher.LightGlueMatcher.match_images

    def match_images(self, *args, **kwargs):
        for key in ("kp1_xy", "kp2_xy", "desc1", "desc2"):
            value = kwargs.get(key)
            if value is not None:
                kwargs[key] = np.asarray(value, np.float32)
        return original(self, *args, **kwargs)

    feature_matcher.LightGlueMatcher.match_images = match_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_dir", type=Path, help="directory holding the slides")
    parser.add_argument("out_dir", type=Path, help="where transforms and QC images go")
    parser.add_argument("--reference", default=None, help="filename stem of the reference slide")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=850,
        help="longest edge used for registration; the memory knob on a 14 GB machine",
    )
    parser.add_argument(
        "--rigid-only",
        action="store_true",
        help="skip non-rigid, which re-expands the image and is the expensive stage",
    )
    parser.add_argument(
        "--keep-staircase",
        action="store_true",
        help="leave the synthetic background in, to show what it does to matching",
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="match on edges rather than colour, which is what works across stains",
    )
    parser.add_argument(
        "--single-matcher",
        action="store_true",
        help="sort and match with the same detector, skipping the rematch pass",
    )
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="rotation and translation only; serial sections are the same size",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=None,
        help="blur by this sigma before the gradient, to suppress cell-level texture "
             "the detector cannot tell apart",
    )
    parser.add_argument(
        "--detector",
        default=None,
        help="feature detector class from valis.feature_detectors, e.g. DiskFD; the "
             "default descriptors are not distinctive on repetitive liver texture",
    )
    parser.add_argument(
        "--check-reflections",
        action="store_true",
        help="re-match on flipped images; BROKEN upstream, see patch_lightglue_dtype",
    )
    args = parser.parse_args()

    from gradient import GradientOD, SmoothGradientOD
    from staircase import NoStaircase
    from valis import registration
    from valis.preprocessing import OD

    # Use one matcher for sorting and matching to avoid empty-match rematch failures.
    matchers = {}
    if args.single_matcher:
        from valis import feature_detectors, feature_matcher

        detector = getattr(feature_detectors, args.detector or "VggFD")
        single = feature_matcher.Matcher(feature_detector=detector())
        matchers = {"matcher": single, "matcher_for_sorting": single}

    # Lock scale for same-size serial sections; pair with reflection checks to avoid flips.
    transformer = {}
    if args.fixed_scale:
        from skimage.transform import EuclideanTransform

        transformer = {"transformer_cls": EuclideanTransform}

    if args.check_reflections:
        patch_lightglue_dtype()

    start = time.time()
    registrar = registration.Valis(
        str(args.slide_dir),
        str(args.out_dir),
        **matchers,
        **transformer,
        # Ovoid sections can match upside down; re-match flipped images and keep the
        # best fit to avoid 180-degree errors.
        check_for_reflections=args.check_reflections,
        reference_img_f=args.reference,
        # keep the reference fixed so its coordinates stay the analysis frame
        align_to_reference=bool(args.reference),
        max_processed_image_dim_px=args.max_dim,
        max_non_rigid_registration_dim_px=args.max_dim,
        # Keep cached unprocessed images at the requested size instead of VALIS's inferred size.
        max_image_dim_px=args.max_dim,
        # non-rigid re-expands the image, so it is the stage that runs out of memory
        non_rigid_registrar_cls=None if args.rigid_only else registration.DEFAULT_NON_RIGID_CLASS,
    )

    if args.rigid_only:
        # VALIS cleanup requires this dict even in rigid-only mode.
        registrar.non_rigid_reg_kwargs = {}

    if args.smooth:
        SmoothGradientOD.sigma = args.smooth
        processor = SmoothGradientOD
    elif args.gradient:
        processor = GradientOD
    elif args.keep_staircase:
        processor = OD
    else:
        processor = NoStaircase

    # bioformats2raw splits brightfield channels; specify the processor explicitly.
    processor_dict = {str(p): processor for p in sorted(args.slide_dir.iterdir())
                      if not p.name.startswith(".")}
    print(f"processor: {processor.__name__} on {len(processor_dict)} slides")

    _rigid, _non_rigid, error_df = registrar.register(processor_dict=processor_dict)

    save_transforms(registrar, args.out_dir / "transforms.json")

    print(f"\nregistered in {time.time() - start:.0f} s")
    print(f"slides: {len(registrar.slide_dict)}")
    print(f"reference: {registrar.reference_img_f}")

    if error_df is not None:
        # one row per slide; the D columns are residual distances in microns
        print("\nerror summary")
        print(error_df.to_string())
        error_df.to_csv(args.out_dir / "registration_error.csv", index=False)

    registration.kill_jvm()


if __name__ == "__main__":
    main()
