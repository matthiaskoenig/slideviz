# slideviz

Visualization for whole-slide liver histology.

Reads whole-slide images and displays them in napari as multiscale layers.

## Setup

```bash
uv sync
export SLIDEVIZ_DATA=/path/to/slides    # or put it in a .env file
```

Python 3.12: napari, VisPy and pylibCZIrw have no wheels for newer versions yet.

`SLIDEVIZ_DATA` is where the slides and their sidecars live; `--data` overrides it 
per run. `SLIDEVIZ_DB` moves the index, which otherwise sits in the user cache 
directory.

## Use

```bash
uv run slideviz-view                                    # browse slides in napari
uv run slideviz-view --no-reindex                       # skip the rebuild, open what is indexed
uv run slideviz-index <dir>                             # rebuild the index
```