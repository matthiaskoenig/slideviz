# ml/

Frozen pathology encoders and linear heads. Separate uv project, runs on `gpu1`.

| File | |
|---|---|
| `extract_embeddings.py` | HepatoBench patches to cached embeddings |
| `train_linear_head.py` | embeddings to classifier, metrics, misclassified list |
| `embed_tiles.py` | mouse slide tiles to cached embeddings |
| `train_necrosis.py` | leave one animal out, `--tag` names the run |
| `predict_slide.py` | one held-out animal's per-tile scores |
| `predict_unlabelled.py` | scores for a slide with no annotation, to show where to look |

The first two are stage 1, HepatoBench. The rest are stage 2, mouse necrosis, and
take their tiles from `slideviz-scripts/export_tiles.py`.

## Setup

    ssh gpu1                       # HU VPN first
    cd ~/Projects/slideviz && git pull
    cd ml && uv sync

## Running

    uv run python extract_embeddings.py \
        --zips /data/michelle/hepatobench/zips \
        --out  /data/michelle/hepatobench/embeddings

    uv run python train_linear_head.py \
        --cache /data/michelle/hepatobench/embeddings

Mouse necrosis, from tiles already on the server:

    uv run python embed_tiles.py \
        --tiles /data/michelle/mouse/tiles \
        --out   /data/michelle/mouse/embeddings

    uv run python train_necrosis.py --cache /data/michelle/mouse/embeddings
    uv run python train_necrosis.py --cache /data/michelle/mouse/embeddings --drop-boundary

    uv run python predict_slide.py \
        --cache  /data/michelle/mouse/embeddings \
        --tiles  /data/michelle/mouse/tiles \
        --animal 281mg_m1 \
        --out    /data/michelle/mouse/predictions
