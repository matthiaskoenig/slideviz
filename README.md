# slideviz

Visualization for whole-slide liver histology.

Reads whole-slide images and displays them in napari as multiscale layers.

## Setup

```bash
uv sync
```

Python 3.12: napari, VisPy and pylibCZIrw have no wheels for newer versions yet.

## Use

```bash
uv run slideviz-view                                    # browse slides in napari
uv run slideviz-index <dir>                             # rebuild the index
uv run slideviz-index --sql "SELECT * FROM slides"      # query it
uv run python scripts/probe_slide.py <file|dir>         # report what is in a slide file
```

## Data layout

```
species_substance_dose_animal_stain.czi
mouse_apap_000mg_m1_he.czi
```

Each slide has a JSON sidecar of the same stem next to it, holding provenance: 
species, substance, dose, animal, stain, modality, and the `serial_block` key 
shared by sections cut from the same block.
`slideviz.catalog` indexes them into a SQLite file for querying, which is derived 
and can be rebuilt at any time.