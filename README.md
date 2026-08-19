# slideviz

Visualization for whole-slide liver histology.

Reads whole-slide images and displays them in napari as multiscale layers.

## Setup

```bash
uv sync
export SLIDEVIZ_DATA=/path/to/slides    # or put it in a .env file
```

## Use

```bash
uv run slideviz-view                                    # browse slides in napari
uv run slideviz-view --data /path/to/slides             # browse slides in napari (sepcific data path)
```