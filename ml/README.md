# ML

Frozen encoder (Lunit ViT-S/8) and linear heads. Own venv, runs on GPU server.

| File | |
|---|---|
| `extract_embeddings.py`, `train_linear_head.py` | HepatoBench baseline |
| `embed_tiles.py`, `stain_norm.py` | slide tiles to cached embeddings (`--patches` keeps the patch descriptions) |
| `train_necrosis.py`, `predict_slide.py`, `predict_unlabelled.py` | tile head, leave one animal out |
| `train_patch_head.py`, `predict_slide_patches.py` | the necrosis map: per-patch head over a whole slide |

Tiles come from `scripts/export_tiles.py` and `scripts/export_field_tiles.py`, labels from
`scripts/patch_labels.py`; `scripts/necrosis_slides.py` turns the maps into calls and areas.

    cd ml && uv sync
    uv run python embed_tiles.py --tiles <tiles> --out <embeddings> --patches
    uv run python train_patch_head.py --embeddings <embeddings> --labels <labels> --out <heads>
    uv run python predict_slide_patches.py --heads <heads> --embeddings <embeddings> \
        --tiles <export_field_tiles.py --slide output> --out <maps>
