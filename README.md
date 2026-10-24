# SlideViz

Whole-slide liver histology in napari: slides, their registration, and the readouts computed from them as layers.

## Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 or 3.13
(3.14 has no `pylibCZIrw` wheel).

```bash
git clone https://github.com/matthiaskoenig/slideviz.git
cd slideviz
uv sync
```

## Data

Point it at a directory of slides (`.czi`, `.ndpi`, OME-TIFF or OME-Zarr) with one JSON sidecar
per slide, named `<stem>.json` or `<stem>.ome.json`:

```bash
echo 'SLIDEVIZ_DATA=/path/to/slides' > .env   # persistent
export SLIDEVIZ_DATA=/path/to/slides          # or per session
```

Readouts are found beside the slides: `predictions/` (necrosis maps, CYP2E1 readouts),
`flatten/` (stain flattening fields) and `stain_reference.json`.

## Use

```bash
uv run slideviz-view                          # browse slides in napari
uv run slideviz-view --data /path/to/slides   # another directory
uv run slideviz-view --no-reindex             # open the existing index

uv run slideviz-index /path/to/slides         # rebuild the index
uv run slideviz-index --schema                # the sidecar JSON Schema
```

The index is derived from the sidecars and rebuilt on every start unless `--no-reindex`.

| Variable | Meaning | Default |
|---|---|---|
| `SLIDEVIZ_DATA` | slides and their sidecars | none, must be set |
| `SLIDEVIZ_DB` | where the index is written | `~/.cache/slideviz/slides.db` |
| `SLIDEVIZ_PREDICTIONS` | readouts | `<data>/predictions` |
| `SLIDEVIZ_STAIN_REFERENCE` | stain statistics the viewer matches slides to | `<data>/stain_reference.json` |

## Layout

| Folder | What | Environment |
|---|---|---|
| `src/slideviz/` | readers, sidecar schema and index, analysis, viewer | this project |
| [`registration/`](registration/README.md) | serial-section registration, one command | own venv |
| [`ml/`](ml/README.md) | frozen encoder, tile and patch heads, necrosis maps | own venv, GPU |
| [`scripts/`](scripts/README.md) | the pipeline steps a reported result depends on | this project |
| [`web/`](web/README.md) | static browser viewer, parked | own venv |
