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

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `SLIDEVIZ_DATA` | Directory of slides and their sidecars | none, must be set |
| `SLIDEVIZ_DB` | Where the index is written | `~/.cache/slideviz/slides.db` |

The index is derived from the sidecars and rebuilt on every start.
