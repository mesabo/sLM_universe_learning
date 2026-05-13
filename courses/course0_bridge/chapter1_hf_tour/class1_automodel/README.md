# Course 0 · Chapter 1 · Class 1 — The Hugging Face ecosystem in 30 minutes

> Goal: understand `AutoModel`, `AutoTokenizer`, the `pipeline()` shortcut, and when to use raw `Trainer` vs a custom training loop. After this class you'll never copy-paste a `from_pretrained` call again — you'll know what each argument means.

---

## Psycho — the mental model

> **One-line takeaway:** Hugging Face is a *catalog plus three power tools*. Learn what each tool does once, and you stop copy-pasting.

When students first open the `transformers` source they see thousands of classes and freeze. The trick is to notice that almost all of them sort into **three concentric layers**:

1. **The hub** — a giant lookup table of (`name → weights + config + tokenizer`). You only need the *name* and the right `Auto*` class. Examples used throughout this course: `sentence-transformers/all-MiniLM-L6-v2`, `HuggingFaceTB/SmolLM2-135M-Instruct`.
2. **The model classes** — `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification`, … each one wires a *task head* on top of the same backbone. Pick the head that matches your job.
3. **The trainer/pipeline** — `pipeline()` for one-line inference, `Trainer` for one-call training, custom loops when you outgrow `Trainer` (Course 1 onwards).

**The four-verb shortcut.** Every line you write does one of *load → tokenize → forward → decode*. Most bugs live at the seams between two consecutive verbs (e.g. "I tokenized but forgot to move the tensors to the same device as the model"). When something breaks, ask: *"which seam am I at?"* — it's almost always faster than re-reading the stack trace.

**Common confusion to head off:** `AutoModel` is **not** the model — it's the *factory*. The thing it returns is the actual `BertModel` (or `LlamaModel`, etc.) under the hood. So an `isinstance(model, AutoModel)` check will fail; that's expected.

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

- [`train.py`](./train.py) — loads a backbone, runs `iterations.n_passes` timed forward passes (after `iterations.warmup` untimed ones), then writes a `metrics.json` via `shared.eval_harness.run_eval`. Latency stats (mean, p50, p95, throughput) land in the result JSON's `extras` block.
- [`configs/default.yaml`](./configs/default.yaml) — backbone, prompt, `iterations` block (defaults to one pass for the cheap-and-fast sanity check), expected band.
- [`run.sh`](./run.sh) — one-line entrypoint. It includes a commented-out slot for `intfloat/multilingual-e5-small` if you want to extend the sweep to a 6th backbone.

### Playing with multiple forward passes and varied input

The default config ships with a 3-prompt list (short / medium / long) and `n_passes=1`. Override `iterations.n_passes` to cycle through the list:

```bash
# Single backbone, 6 passes (each prompt seen twice)
python train.py --config configs/default.yaml \
    iterations.n_passes=6 iterations.warmup=1
```

Per-pass `latency_ms` and a summary (mean / p50 / p95 / throughput) print to the log. The result JSON's `extras.prompt_indices[]` and `extras.latencies_ms[]` line up — you can compute per-prompt-length latency yourself.

Use `prompts: [...]` (preferred) or the legacy `prompt: "..."` (single string). If both are set, `prompts` wins. To benchmark on your own prompts, override at the CLI:

```bash
python train.py --config configs/default.yaml \
    'prompts=["foo","bar","baz"]' iterations.n_passes=12
```

Expect mean latency ~1–5 ms on a warm GPU for MiniLM, ~10–30 ms for SmolLM2-360M. The first pass is always slower (cuDNN init); set `warmup>=1` to discard it.

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
| `n_passes_ran` | 1 | 100000 | Sanity: the iteration loop actually ran |
| `mean_latency_ms` | 0 | 600000 | Permissive — machines vary. The interesting number is in `extras.p50_latency_ms` and `throughput_passes_per_s`. |

If `eval.py` exits non-zero, your install is broken — fix the env before moving on.
