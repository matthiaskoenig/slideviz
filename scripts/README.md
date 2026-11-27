# Scripts

The pipeline steps a reported result depends on.

| Script | |
|---|---|
| `export_polygons.py`, `export_tiles.py`, `patch_labels.py` | annotations to training tiles and patch labels |
| `export_field_tiles.py` | overlapping tiles over a whole slide or a field |
| `necrosis_lumen.py`, `necrosis_slides.py` | lumen masks, then the necrosis map's calls and areas |
| `flatten_fields.py`, `cyp2e1_readout.py` | stain flattening fields, then the CYP2E1 readout |
| `landmarks.py` | hand landmarks for registration error |
| `mesh_inputs.py` | outline, lumina with their zonation classes, necrosis map and CYP2E1 map of a field or section on one grid, for `mesh/` |
| `write_scene.py` | scene.json and the zonation, CYP2E1 and necrosis maps (Zarr) of tagged polygons (`--polygons`, `mesh/polygons.py` output, `--size-um SMALL:LARGE`), meshed by slidesim |
| `wall_class_check.py` | portal and central lumen classes on dosed slides: counts by class, lumina at the necrosis |
