# Registration

Serial sections registered onto the H&E: outline rigid (`outline_register.py`), then DeepFlow
non-rigid (`nonrigid_register.py`). `pipeline.py` runs both for every block and writes the
transform and the field into sidecar copies under `<run>/slides/`.

Own venv (`valis-wsi` needs `numpy<2`):

    cd registration && uv sync

From the repo root:

    uv run python registration/pipeline.py all --slides <slides> --out <run> --jobs 10
