# slideviz

Visualization for whole-slide liver histology.

Reads whole-slide images and displays them in napari as multiscale layers.

## Setup

```bash
uv sync
```

Python 3.12: napari, VisPy and pylibCZIrw have no wheels for newer versions yet.

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