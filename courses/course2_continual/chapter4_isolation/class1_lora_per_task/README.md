# Course 2 · Chapter 4 · Class 1 — Parameter isolation: LoRA per task

> Goal: stop catastrophic forgetting **by construction** — give each task its own LoRA adapter (and its own classification head) over a frozen shared backbone. At eval time, swap to the right adapter per task. BWT should be ≈ 0 because the parameters that defined Task A are never touched while learning Task B.

---

## Psycho — the mental model

> **One-line takeaway:** stop arguing about *how much* Task B is allowed to edit Task A's parameters — just *don't share the parameters in the first place*. BWT becomes zero because there's no mechanism for forgetting to occur.

Replay (ch2) and EWC (ch3) both let Task B's training edit the same weights that learned Task A — they only differ in *how much* they constrain those edits. Parameter isolation gives up on that argument entirely. The base encoder stays frozen forever (it's the *shared knowledge*); each task gets its own private parameters (the *task-specific bits*):

- Its own LoRA adapter (small, ~0.3% of params, plugged into the attention projections).
- Its own classification head (~1% of params, mapping pooled output → labels).

When you evaluate Task A, you swap to adapter_A + head_A. When you evaluate Task B, you swap to adapter_B + head_B. The base encoder's pretrained features are shared across tasks (cheap, no forgetting); the task-specific deltas are private (no interference). Forgetting is mathematically impossible because the parameters that defined Task A are never touched while learning Task B.

The cleanness comes at a real cost: there's no positive transfer either. EWC and replay let Task B's gradients influence Task A's weights, which sometimes *helps* Task A (the model finds a representation that serves both). Isolation forbids that by construction.

**Pros:**
- BWT ≈ 0 *by design* — there's no mechanism for Task B to overwrite Task A.
- Tiny per-task storage (an adapter is a few MB).
- Hot-swap at inference: one base loaded, multiple adapters, microsecond switching.

**Cons:**
- You need a **task router** at inference (which adapter for this query?). For a known small set of tasks, that's trivial; for open-ended deployment, it's a research problem.
- No positive transfer either — Task B's training can't *help* Task A, even when sharing knowledge would be beneficial.
- Storage scales linearly with number of tasks (still cheap, just not free).

## Academic — what's happening

Let $\theta_b$ be the (frozen) base parameters and $\phi_t$ be the trainable task-specific parameters (LoRA matrices + classification head) for task $t$. The loss for task $t$ optimizes only $\phi_t$:

$$\min_{\phi_t} \; \mathbb{E}_{(x,y) \sim D_t}\!\left[\,-\log p(y \mid x; \,\theta_b, \phi_t)\,\right]$$

The base $\theta_b$ stays exactly at its pretrained values throughout the entire course. Catastrophic forgetting requires $\phi_t$ updates to leak into another task's parameters — which is impossible when $\phi_A$ and $\phi_B$ are disjoint.

This is the modern PEFT-era answer to the continual-learning problem. The PEFT library makes it almost trivial:

```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    modules_to_save=["classifier"],   # the head is per-task too
    task_type="SEQ_CLS",
)
model = get_peft_model(base, cfg, adapter_name="ag_news")

# Train Task A. Then:
model.add_adapter("emotion", cfg)
model.set_adapter("emotion")
# Train Task B. The "ag_news" adapter is untouched.

# Eval:
model.set_adapter("ag_news"); evaluate(task_A)
model.set_adapter("emotion"); evaluate(task_B)
```

References:
- [Hu et al., *LoRA: Low-Rank Adaptation* (ICLR 2022)](https://arxiv.org/abs/2106.09685) — the underlying method
- [PEFT — multi-adapter loading](https://huggingface.co/docs/peft/developer_guides/troubleshooting#using-multiple-adapters)
- [Pfeiffer et al., *AdapterFusion* (EACL 2021)](https://arxiv.org/abs/2005.00247) — earliest "adapter per task" continual-learning result
- [Wang et al., *Orthogonal Subspace Learning* (NeurIPS 2023)](https://arxiv.org/abs/2310.14152) — keeps adapters from interfering even when not strictly disjoint

## Engineering — what the code does

[`train.py`](./train.py):

1. **Stage 0** — eval both tasks with no adapter active (just the base + a freshly-initialized head shared across tasks). Both ~25% (random for 4-class).
2. **Stage 1** — wrap the base with `peft.get_peft_model` using `adapter_name="ag_news"`. `modules_to_save=["classifier"]` makes the head per-adapter. Train Task A (only the LoRA matrices and the AG-News head update; base stays frozen). Re-eval both tasks (Task B uses the AG-News adapter+head, since it's the only one that exists yet).
3. **Stage 2** — add a second adapter `model.add_adapter("emotion", lora_config)`. `set_adapter("emotion")` activates it. Train Task B (only the Emotion LoRA matrices and head update). Re-eval each task with its own adapter.

The expected band asserts BWT is essentially 0 (`|BWT| <= 0.05` — wider than it strictly needs to be to absorb floating-point and head-init noise).

### Gotchas
- **`modules_to_save=["classifier"]` is essential** — without it, the classification head is shared across adapters and Task B's training rewrites the head used to classify Task A.
- The "classifier" name is architecture-specific. For BERT-style encoders (`AutoModelForSequenceClassification`) the head module is named `classifier`. For other architectures, find the head with `print(model)` and put its name here.
- `set_adapter("name")` is needed before BOTH training and eval. Forgetting to swap at eval time is the most common bug.
- `model.add_adapter` with a `LoraConfig` instance works, but you can also load adapters from disk with `model.load_adapter("path/to/saved", "name")`.

## Research — open questions

- Compare ch4 BWT to ch3 (EWC, ~−0.24) and ch2 (replay r=0.25, ~+0.02). Isolation should beat both in BWT — at the cost of zero positive transfer between tasks.
- What if you allow the base encoder to be **slightly** trainable (LoRA on the base too, shared across tasks)? You get partial sharing. Where does BWT degrade as you increase the shared trainable budget?
- Inference-time routing: train a third tiny "router" head that predicts which adapter to use for an incoming query. How much accuracy do you lose?
- How does this generalize to 5+ tasks? At what point does adapter-storage become a real cost?

---

## How to run

```bash
bash courses/course2_continual/chapter4_isolation/class1_lora_per_task/run.sh
```

Smoke mode by default — runs in ~1–2 min.

## How to verify

`results/full/<backbone>/course2_continual/chapter4_isolation_class1_lora_per_task/two_task/iso-r16.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `avg_accuracy` | 0.50 | 1.0 | Mean across both tasks at the final stage (each evaluated with its own adapter) |
| `acc_A_after_A` | 0.55 | 1.0 | Sanity: A really was learned |
| `acc_B_after_B` | 0.35 | 1.0 | Sanity: B really was learned |
| `acc_A_after_B` | 0.55 | 1.0 | The headline: A retained essentially perfectly (== `acc_A_after_A`) |
| `BWT` | -0.05 | 0.05 | Near zero **by construction** — this is the lesson |
| `n_adapters` | 2 | 2 | Sanity: both adapters created |
| `trainable_ratio_pct` | 0.0 | 5.0 | Sanity: most params frozen |

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections (Psycho / Academic / Engineering / Research) are present and ≥ 2 paragraphs each.
- [ ] Every reference link points at an official source (paper / HF doc / repo) where one exists.
- [ ] `train.py` and `eval.py` contain no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written by `eval.py`.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises (warm-up / apply / stretch).
- [ ] Result JSON path matches the layout `results/full/<backbone>/<course>/<class>/<task>/<method>.json`.
- [ ] At least one smoke-mode run completed end-to-end and the metric band passes.
- [ ] Linked from the parent course `README.md` table.
