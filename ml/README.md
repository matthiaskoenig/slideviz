# ml/

Frozen pathology encoders and linear heads. Separate uv project, runs on `gpu1`.

| File | |
|---|---|
| `extract_embeddings.py` | patches to cached embeddings |
| `train_linear_head.py` | embeddings to classifier, metrics, misclassified list |

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
