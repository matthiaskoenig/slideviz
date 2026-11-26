# web

Static [Vitessce](https://vitessce.io) page for the OME-Zarr slides, served as plain files behind
Caddy. Parked; the viewer is napari. Own venv (`vitessce[all]` caps `dask`).

    cd web && uv sync
    uv run python build_config.py    # -> dist/config.json, dist/index.html

`SLIDES` in `build_config.py` lists the stores, `BASE_URL` the host. Vitessce is served from the
site's own origin (`vendor/`, gitignored, from the npm tarball of 3.9.11). Deploy `vendor/`, `dist/`
and the `.zarr` stores to the site root with `tar | ssh`. A registered pair shows unaligned: the config
takes no affine.
