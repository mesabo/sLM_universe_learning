# Notes — Course 0 · ch 1 · class 1 (HF ecosystem tour)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | Backbone | hidden | params_m | forward_ok | Notes |
|---|---|---|---|---|---|
| default | sentence-transformers/all-MiniLM-L6-v2 | 384 | 22 |  |  |
| commented option | intfloat/multilingual-e5-small |  | 118 |  | Uncomment in `run.sh` when ready to test multilingual retrieval. |
|  |  |  |  |  |  |

## Exercises

### 1. Warm-up — read the config

For `HuggingFaceTB/SmolLM2-135M-Instruct`, I found:
- `hidden_size`: 576
- `num_attention_heads`: 9
- `num_key_value_heads`: 3
- `max_position_embeddings`: 8192

Why `num_key_value_heads != num_attention_heads`:
This usually means grouped-query attention. Multiple attention heads share fewer key/value heads to reduce memory and speed up inference.

### 2. Apply — swap to BGE-small

`BAAI/bge-small-en-v1.5` reported a larger `hidden_size` than MiniLM.

Why it is larger even though both are tiny:
Parameter count is not controlled by hidden size alone. Layer count, attention structure, FFN width, vocab, and pooling setup also matter, so two “small” models can still have different hidden sizes.

### 3. Stretch — register a new backbone

I added:
- `intfloat/multilingual-e5-small` to `configs/backbones.yaml`

What changed:
- Registered the model as `kind: sentence-encoder`
- Used `pooling: mean`
- Kept `dtype: float32`

What would need to change in `shared/backbones.py` for a retrieval-tuned encoder:
The current `sentence-encoder` path in [backbones.py](/home/Aboya_25R9803/projects/extra/learning/learning_slm_claude/shared/backbones.py) is enough for this model.
If we wanted a distinct retrieval-tuned kind, we would probably add:
- a new `kind` such as `retrieval-encoder`
- query/document formatting rules like `query:` and `passage:`
- retrieval-specific defaults for pooling or normalization
- retrieval-specific eval assumptions in downstream code
