# Exercises — Course 0 · ch1 · class 1

Three graded exercises. Each takes 5–20 min. The "stretch" one is optional but the most valuable.

## 1. Warm-up — read the config

Open `.cache/huggingface/hub/.../config.json` for `HuggingFaceTB/SmolLM2-135M-Instruct` after `run.sh` completes. Find:

- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`
- `max_position_embeddings`

> Why does `num_key_value_heads != num_attention_heads`? (Answer in `NOTES.md`.)

## 2. Apply — swap to `BAAI/bge-small-en-v1.5`

Run `train.py` with `backbone=BAAI/bge-small-en-v1.5`. Compare the printed `hidden_size` to MiniLM's. Why is it larger even though both are tiny?

## 3. Stretch — register a new backbone

Add a 6th entry to `configs/backbones.yaml` (e.g. `intfloat/e5-small-v2`). Run this class against it. What in `shared/backbones.py` would need to change if the new backbone were a *retrieval-tuned* encoder rather than a sentence-encoder?
