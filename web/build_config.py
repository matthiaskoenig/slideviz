"""Build a static Vitessce viewer for the APAP OME-Zarr slides.

Writes dist/config.json and dist/index.html. Both are static; the slides
themselves are served as OME-Zarr from BASE_URL.
"""

import json
import os
from pathlib import Path

from vitessce import (
    CoordinationLevel as CL,
)
from vitessce import (
    VitessceConfig,
    get_initial_coordination_scope_prefix,
    hconcat,
)

BASE_URL = "https://slide.michelle-elias.de"
OUT_DIR = Path(__file__).parent / "dist"

# Where the sidecar JSONs live. They sit beside the Zarr stores and carry the
# VALIS matrix, so a re-registration is picked up by rebuilding.
SIDECAR_DIR = Path(
    os.environ.get(
        "SLIDEVIZ_ZARR_DIR",
        "/home/michelle/Projects/image-analysis/images/mouse/APAP_zarr",
    )
)

# label, store path, sidecar. The store path ends in /0 because bioformats2raw
# writes a collection: series 0 is the slide, 1 and 2 are the scanner's label
# and macro thumbnails.
SLIDES = [
    ("H&E", "mouse_apap_500mg_m2_he.zarr/0", "mouse_apap_500mg_m2_he.json"),
    ("CYP2E1", "mouse_apap_500mg_m2_cyp2e1.zarr/0", "mouse_apap_500mg_m2_cyp2e1.json"),
]


def coordinate_transformations(sidecar):
    """Return None: config transforms cannot express this pair's rotation."""
    return


def build(slides):
    vc = VitessceConfig(schema_version="1.0.18", name="APAP mouse liver")
    dataset = vc.add_dataset(name="500 mg/kg m2")

    for name, path, sidecar in slides:
        # fileUid ties each coordination scope below to one specific file.
        # Without it both layers address the same image and stack invisibly.
        transform = coordinate_transformations(sidecar)
        dataset = dataset.add_file(
            file_type="image.ome-zarr",
            url=f"{BASE_URL}/{path}",
            coordination_values={"fileUid": name},
            options=(
                {"coordinateTransformations": transform} if transform else None
            ),
        )

    spatial = vc.add_view("spatialBeta", dataset=dataset)
    controller = vc.add_view("layerControllerBeta", dataset=dataset)

    # Explicitly set brightfield RGB.
    vc.link_views_by_dict(
        [spatial, controller],
        {
            "imageLayer": CL([
                {
                    "fileUid": name,
                    "photometricInterpretation": "RGB",
                }
                for name, _, _ in slides
            ])
        },
        scope_prefix=get_initial_coordination_scope_prefix("A", "image"),
    )

    vc.layout(hconcat(spatial, controller, split=(3, 1)))
    return vc


PAGE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>APAP mouse liver</title>
    <style>html,body,#root{margin:0;height:100%%;background:#111}</style>
    <!-- Vitessce comes from this origin rather than a CDN. esm.sh injects a
         Node `process` polyfill, which makes the bundle's isNode check true and
         then hits a bare `module` reference meant for worker_threads. -->
    <script type="importmap">
    {
      "imports": {
        "react": "https://esm.sh/react@18.3.1?dev=false",
        "react-dom": "https://esm.sh/react-dom@18.3.1?dev=false",
        "react-dom/client": "https://esm.sh/react-dom@18.3.1/client?dev=false"
      }
    }
    </script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module">
      import React from 'react';
      import { createRoot } from 'react-dom/client';
      import { Vitessce } from '../vendor/dist/index.min.js';

      const config = await (await fetch('./config.json')).json();
      createRoot(document.getElementById('root')).render(
        React.createElement(Vitessce, { config, theme: 'dark', height: window.innerHeight })
      );
    </script>
  </body>
</html>
"""

VITESSCE_JS = "3.9.11"

if __name__ == "__main__":
    vc = build(SLIDES)
    OUT_DIR.mkdir(exist_ok=True)

    config = vc.to_dict(base_url=BASE_URL)
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    (OUT_DIR / "index.html").write_text(PAGE % {"v": VITESSCE_JS})

    print(f"wrote {OUT_DIR}/config.json and index.html")
    print(f"datasets: {len(config['datasets'])}, views: {len(config['layout'])}")
