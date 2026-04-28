# Course 0 · Chapter 1 · Class 1 — The Hugging Face ecosystem in 30 minutes

> Goal: understand `AutoModel`, `AutoTokenizer`, the `pipeline()` shortcut, and when to use raw `Trainer` vs a custom training loop. After this class you'll never copy-paste a `from_pretrained` call again — you'll know what each argument means.

---

## Psycho — the mental model

Think of Hugging Face as **three concentric layers**:

1. **The hub** — a giant lookup table of (`name → weights + config + tokenizer`). You only need the *name* and the right `Auto*` class. Examples in this course: `sentence-transformers/all-MiniLM-L6-v2`, `HuggingFaceTB/SmolLM2-135M-Instruct`.
2. **The model classes** — `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification`, … each one wires a *task head* on top of the same backbone.
3. **The trainer/pipeline** — `pipeline()` for one-line inference; `Trainer` for one-call training; custom loops when you outgrow `Trainer` (Course 1 onwards).

**Mental shortcut:** every line you write is one of *load → tokenize → forward → decode*. Most bugs are at the seams between two of those.

## Academic — what's actually happening

A model on the hub is just three artifacts:

- `config.json` — architecture hyperparameters (hidden size, layers, attention heads, vocab size, special-token ids).
- `tokenizer.json` (or `vocab.txt` + `merges.txt`) — the tokenizer state.
- `model.safetensors` (or `pytorch_model.bin`) — the parameters.

`AutoModel.from_pretrained(name)` does:

1. Resolve `name` → repo on the hub (or local cache).
2. Read `config.json` → look up the `architectures` field → import the matching Python class (e.g. `BertModel`, `LlamaForCausalLM`).
3. Build the module with that config, then load weights from `model.safetensors`.

So `AutoModel` is a **factory + dispatcher**, not a model itself. References:
- [Hugging Face — Auto classes](https://huggingface.co/docs/transformers/model_doc/auto)
- [Hugging Face — Tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)

## Engineering — the code in this class

Everything goes through `shared.backbones.load_backbone(name)` so the rest of the course never touches `from_pretrained` directly. This class shows what `load_backbone` does *under the hood* — it's the only place we deliberately call `AutoModel`.

Files:

- [`train.py`](./train.py) — actually a no-op for this class; it loads a backbone, prints its parameter count and a small forward-pass sanity check, then writes a `metrics.json` via `shared.eval_harness.run_eval`.
- [`configs/default.yaml`](./configs/default.yaml) — backbone, prompt, expected output-shape band.
- [`run.sh`](./run.sh) — one-line entrypoint.

### Gotchas
- `AutoModel` returns the *backbone*, not a task head. For classification you want `AutoModelForSequenceClassification`; for generation, `AutoModelForCausalLM`.
- Encoder models (BERT-style) return *contextual embeddings*; decoder models return *next-token logits*. We expose this via `Backbone.kind` in [`shared/backbones.py`](../../../../shared/backbones.py).
- Tokenizers are *not* models — they're separate objects with separate caches. Always load both via the same `name` to avoid mismatch.

## Research — open questions / extensions

- The hub has multiple revisions per model (branches, tags). When does pinning `revision="…"` matter? (Hint: it always matters once a result is in a paper.)
- `pipeline()` defaults can change between `transformers` versions. Find the current default for `text-classification` and what happens when you swap models silently.
- Try loading the same model with `torch_dtype=torch.bfloat16` vs `torch.float32` and measure peak GPU memory. Why isn't the ratio exactly 2×?

---

## How to run

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"  # project root
bash courses/course0_bridge/chapter1_hf_tour/class1_automodel/run.sh
```

The first run downloads ~22 MB into `.cache/huggingface/`.

## How to verify

`eval.py` writes `results/full/<backbone>/course0_bridge/chapter1_hf_tour_class1_automodel/sanity/load.json`. The expected band is:

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `forward_ok` | 1 | 1 | The forward pass returned a non-empty tensor |
| `hidden_size_ok` | 1 | 1 | Reported hidden size > 0 |

If `eval.py` exits non-zero, your install is broken — fix the env before moving on.
