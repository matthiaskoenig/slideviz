# web: the browser viewer

Static [Vitessce](https://vitessce.io) viewer for the APAP OME-Zarr slides.
The build writes two small files; everything else is the OME-Zarr stores and a
vendored copy of Vitessce, both served as plain static files.

## Its own environment

Separate uv project. `vitessce[all]` pins
`ome-zarr==0.15.0`, which caps `dask<=2026.1.1`, while slideviz needs
`dask>=2026.7.1`. The two resolve together only by holding slideviz's dask six
months back and adding 47 packages,.

## Build

Run from this directory, so uv picks this project:

    cd web && uv run python build_config.py    # -> dist/config.json, dist/index.html

`build_config.py` reads the registration matrix from the sidecar JSONs beside
the Zarr stores (`SIDECAR_DIR`), so a re-registration is picked up by rebuilding.
Point `BASE_URL` at your own host before deploying.

Slide URLs must end in `.zarr/0`. `bioformats2raw` writes a collection where
series 0 is the slide and series 1 and 2 are the scanner's label and macro
thumbnails.

## vendor/ (gitignored, 20 MB)

Vitessce is served from our own origin. esm.sh injects a Node
`process` polyfill, which makes the bundle's `isNode` check true and then hits a
bare `module` reference meant for `worker_threads`. To fetch it:

    curl -sL https://registry.npmjs.org/vitessce/-/vitessce-3.9.11.tgz -o v.tgz
    mkdir -p vendor && tar -xzf v.tgz -C vendor --strip-components=1 package/dist

## Server

Any static file server works, given range requests and the right MIME type for
`.js`. The deployed setup is Caddy on a Hetzner VPS, which gets HTTPS
automatically once the DNS A record points at the box.

Layout under the site root:

    /                       the .zarr stores, served as directories of chunks
    /vendor/dist/           the Vitessce bundle
    /viewer/                config.json and index.html

`/etc/caddy/Caddyfile`:

    slides.example.com {
        root * /var/www/slides

        # Only needed while testing against viewers on other origins, such as
        # ome-ngff-validator or Avivator. The bundled viewer is same-origin.
        header {
            Access-Control-Allow-Origin  "*"
            Access-Control-Allow-Methods "GET, HEAD, OPTIONS"
            Access-Control-Allow-Headers "Range, Content-Type"
            Access-Control-Expose-Headers "Content-Range, Content-Length, Accept-Ranges"
        }
        @options method OPTIONS
        respond @options 204

        # Zarr chunks are already blosc-compressed. Only the metadata is worth gzipping.
        @meta path *.zattrs *.zgroup *.zarray *.json *.xml
        encode @meta gzip

        file_server browse
    }

Then `sudo systemctl reload caddy`.

`file_server browse` turns on directory listing, which
is useful while checking paths and should come off before sharing the link.

Caddy sends `Accept-Ranges: bytes` on static files by default. Exposing
`Content-Range` matters because browsers hide non-safelisted response headers on
cross-origin reads.

## Deploy

    tar -cf - vendor | ssh HOST 'tar -C /var/www/slides -xf -'
    cd dist && tar -cf - config.json index.html | ssh HOST 'tar -C /var/www/slides/viewer -xf -'

One slide is ~4700 files, so `rsync` pays an ssh round trip per file and crawls.
Stream it as one `tar` instead. The chunks are already compressed, so `-z` costs
CPU for nothing:

    tar -C /path/to/APAP_zarr -cf - mouse_apap_500mg_m2_he.zarr \
      | ssh HOST 'tar -C /var/www/slides -xf -'


## Known limits

- **No affine transform.** Vitessce 3.9.11 validates
  `coordinateTransformations` against `identity`, `translation` and `scale`
  only, so a registered pair cannot be shown aligned from the config. The
  runtime does handle `type: "affine"`, but only from the store's own NGFF
  metadata. Either bake the matrix into the pixels or write the affine into
  `.zattrs`.
- **No slide picker.** A Vitessce config is static, one JSON per dataset, so
  browsing N slides currently means generating N pages.
