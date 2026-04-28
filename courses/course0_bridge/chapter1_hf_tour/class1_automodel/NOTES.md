# Notes — Course 0 · ch 1 · class 1 (HF ecosystem tour)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | Backbone | hidden | params_m | forward_ok | Notes |
|---|---|---|---|---|---|
| default | sentence-transformers/all-MiniLM-L6-v2 | 384 | 22 |  |  |
|  |  |  |  |  |  |

## Exercises

### 1. Warm-up — read the config

(Find `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `max_position_embeddings` for SmolLM2-135M. Why does `num_key_value_heads != num_attention_heads`?)

### 2. Apply — swap to BGE-small

(Compare hidden_size between MiniLM and BGE-small. Why is BGE-small larger even though both are tiny?)

### 3. Stretch — register a new backbone

(What changed in `configs/backbones.yaml`? What in `shared/backbones.py` would need to change for a non-sentence-encoder retrieval backbone?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
