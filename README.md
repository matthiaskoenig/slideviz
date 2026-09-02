# SlideViz

Visualization for whole-slide liver histology.

Reads whole-slide images and displays them in napari as multiscale layers.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/matthiaskoenig/slideviz.git
cd slideviz
uv sync
```

Point it at a directory holding the slides **and their JSON sidecars**:

```bash
echo 'SLIDEVIZ_DATA=/path/to/slides' > .env   # persistent, any terminal
export SLIDEVIZ_DATA=/path/to/slides          # or per-session
```

## Use

```bash
uv run slideviz-view                                 # browse slides in napari
uv run slideviz-view --data /path/to/slides          # override the configured path

uv run slideviz-view --no-reindex                    # skip the rebuild

uv run slideviz-index /path/to/slides                # rebuild the index alone
uv run slideviz-index --sql "SELECT file FROM slides WHERE dose_mg_per_kg > 200"
uv run slideviz-index --schema                       # the sidecar JSON Schema
```

## Registration

    cd registration && uv sync                       
    uv run python registration/register.py 375mg_m1

Converts the block's slides to OME-TIFF, runs VALIS, checks the error against a
500 µm limit, writes the transform into the sidecars, and reports where the TIFFs
can be archived. Above the limit nothing is written.

Registration is a separate uv project with its own venv: `valis-wsi` needs
`numpy<2`, this project needs `numpy>=2`. They meet through `transforms.json` on
disk: `valis_register.py` writes it to the run's output directory
(`valis_runs/out_<block>/transforms.json`), `slideviz.data.registration` reads it and
copies the matrix into the sidecars. See
[`registration/README.md`](registration/README.md).

| Script | Env | Role |
|---|---|---|
| `registration/register.py` | slideviz | entry point |
| `registration/valis_register.py` | registration | the VALIS call |

## Browser viewer

    cd web && uv run python build_config.py

Builds a static [Vitessce](https://vitessce.io) page for the converted OME-Zarr
slides, served as plain static files behind Caddy. Setup, deployment and the
current limits are in [`web/README.md`](web/README.md).

Also a separate uv project: `vitessce[all]` pins `ome-zarr==0.15.0`, which caps
`dask<=2026.1.1`, and this project needs `dask>=2026.7.1`. `build_config.py`
never imports slideviz; it reads the sidecars by path.

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `SLIDEVIZ_DATA` | Directory of slides and their sidecars | none, must be set |
| `SLIDEVIZ_DB` | Where the index is written | `~/.cache/slideviz/slides.db` |
| `SLIDEVIZ_ZARR_DIR` | Sidecars the `web/` build reads | the mouse `APAP_zarr` path |

The index is derived from the sidecars and rebuilt on every start.

## Python version

The project runs on **3.13**. 3.14 is blocked by
[pylibCZIrw](https://pypi.org/project/pylibczirw/#files), the `.czi` reader,
which ships no cp314 wheel (its metadata claims 3.14 support, so resolution
succeeds and only the install fails, on a source build needing CMake).
