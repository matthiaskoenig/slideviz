# registration

VALIS registration of serial sections.

Separate venv: `valis-wsi` needs `numpy<2`, slideviz needs `numpy>=2`.

## Setup

    cd registration && uv sync    # ~5.6 GB, mostly torch

## Run

Via the entry point in the parent project:

    uv run python scripts/register.py 375mg_m1

Directly:

    uv run python valis_register.py <slide_dir> <out_dir> \
        --reference mouse_apap_375mg_m1_he.ome.tiff --rigid-only --gradient

`--gradient` matches on Sobel edges instead of colour; H&E and DAB share no
colour features. `--rigid-only` skips the non-rigid stage, which OOMs on 16 GB RAM.

## Output

Into `<out_dir>` (in practice `valis_runs/out_<block>/`):
`transforms.json` (3x3 affine per slide, maps moving to reference in x=col,
y=row pixels), `registration_error.csv`, and QC overlaps in `slides/overlaps/`.
`scripts/register.py` reads `transforms.json` back and writes the matrix into
the slide sidecars.

The matrix is fitted by pushing a point grid through `warp_xy` and solving least
squares. Residual ~1e-11 px. Do not read `slide.M`, it is relative to VALIS's
internal cropping.

## Reading the result

Good runs: roughly 13 to 31 µm, scale within ~1.5% of 1.0. Negative determinant 
means a reflection, possibly a section mounted face down.
