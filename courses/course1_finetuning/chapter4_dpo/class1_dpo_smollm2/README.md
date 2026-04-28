# Course 1 · Chapter 4 · Class 1 — Direct Preference Optimization (DPO) on SmolLM2

> Goal: take SmolLM2-135M-Instruct (already SFT-tuned by HuggingFaceTB) and nudge it further with **preference pairs** instead of demonstration data. After this class you'll know what `(prompt, chosen, rejected)` triples buy you over plain SFT, and you'll have implemented the simplest of the post-SFT alignment recipes via TRL's `DPOTrainer`.

---

## Psycho — the mental model

SFT (chapter 1 class 2) teaches the model "here's a good response, mimic it". The implicit ranking is "this response = good, everything else = unspecified". DPO teaches with **comparisons**: "for this prompt, response A is preferred over response B — increase the probability of A relative to B". The training signal is *contrastive*, not *imitative*.

Why does this matter?

- Many post-deployment improvements come from preference data — annotators rank pairs of model outputs, not write demonstrations from scratch.
- Comparisons are cheaper to collect and lower-noise than demonstrations.
- The model learns to prefer certain *styles* and *behaviors*, not just to memorize specific completions.

DPO is the simplest of the modern alignment methods (vs. PPO+RLHF or KTO/IPO/ORPO variants) — no separate reward model, no RL loop. You give it `(prompt, chosen, rejected)` triples and a reference model, and it directly adjusts log-probabilities so chosen rises relative to rejected, while staying close to the reference.

## Academic — what's happening

DPO (Rafailov et al., NeurIPS 2023) derives a closed-form loss from RLHF's KL-constrained reward objective. Let $\pi_\theta$ be the policy (the model we're training) and $\pi_\text{ref}$ the reference (a frozen copy of the same SFT model). The DPO loss for one preference pair $(x, y_w, y_l)$ — prompt $x$, chosen $y_w$, rejected $y_l$ — is:

$$
\mathcal{L}_\text{DPO}(\theta) = -\log \sigma\!\left(\beta \cdot \log\frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} \;-\; \beta \cdot \log\frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)
$$

where $\beta$ controls how far the policy can drift from the reference (default 0.1). The two log-ratios are the **implicit rewards** $\hat{r}_\theta(x, y) = \beta \log(\pi_\theta(y|x) / \pi_\text{ref}(y|x))$. Minimizing the loss makes $\hat{r}_\theta(x, y_w) > \hat{r}_\theta(x, y_l)$ on average.

Standard things TRL's `DPOTrainer` logs while training:

- `rewards/chosen` — mean implicit reward on chosen completions
- `rewards/rejected` — mean implicit reward on rejected completions
- `rewards/margins` — `chosen - rejected` (positive = the model prefers chosen)
- `rewards/accuracies` — fraction of pairs where chosen reward > rejected reward

References:
- [Rafailov et al., *Direct Preference Optimization* (NeurIPS 2023)](https://arxiv.org/abs/2305.18290) — the DPO paper
- [TRL — `DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer)
- [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) — the small (12k) Parquet preference dataset we use here
- [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) — the SFT dataset SmolLM2-Instruct was trained on (chapter 1 class 2)

## Engineering — what the code does

[`train.py`](./train.py):

1. Loads SmolLM2-135M-Instruct twice — once as the trainable policy, once as the frozen reference. Memory cost ≈ 2× one model (~540 MB on GPU for SmolLM2-135M).
2. Loads `Intel/orca_dpo_pairs` (Parquet, 12k pairs), caps to `limits[mode].train` rows, holds out the last `dataset.eval_holdout` for eval.
3. Renames the dataset's `(question, chosen, rejected)` to TRL's expected `(prompt, chosen, rejected)`. The optional `system` field is prepended into the prompt with the chat template.
4. Calls `trl.DPOTrainer` with `DPOConfig(beta=..., max_length=..., learning_rate=..., ...)`. TRL handles all the log-prob math.
5. Reports `train_loss_final`, `eval_loss`, and the final `rewards/{chosen,rejected,margins,accuracies}` from the trainer's log history. Persists a result JSON via `shared.eval_harness.run_eval`.

The metric band asserts the model actually learned to prefer chosen — `rewards_margin > 0` and `rewards_accuracy > 0.5`.

### Gotchas
- **DPO LR is much lower than SFT LR.** Default `5e-6` here; using SFT-scale `2e-5` typically destabilizes training. Bump to `1e-5` only with caution.
- **`beta` controls drift from the reference.** Lower `beta` (e.g. 0.01) lets the policy move further; higher `beta` (e.g. 0.5) keeps it close. 0.1 is the canonical default.
- **The reference model is frozen.** If you accidentally pass `model = ref_model = same_object` and the model trains, the reference drifts too and the loss collapses. The `ref_model` should be an independent fp32 / bf16 copy in eval mode.
- **`max_length` matters.** Pairs that exceed `max_length` get truncated; if truncation cuts off the divergence point between chosen and rejected, the gradient signal is noisy.

## Research — open questions / extensions

- Sweep `beta ∈ {0.01, 0.1, 0.5}`. Plot `rewards/accuracies` vs `eval_loss`. Where's the trade-off between drift and signal?
- Replace DPO with **ORPO** (Hong et al., 2024) — a single-stage alternative that doesn't need a reference model. TRL ships `ORPOTrainer`. Same dataset, same metric definitions; how does the loss landscape compare?
- Try a different preference dataset: `argilla/distilabel-intel-orca-dpo-pairs` is a re-curated variant. Same prompts, different (chosen, rejected) selections — does the model end up preferring different completions?
- Combine with LoRA (chapter 2): `DPOTrainer` accepts a PEFT model and uses the base as the reference automatically (no need to load a second copy). Saves ~half the GPU memory. How does eval loss compare to full-FT DPO?

---

## How to run

```bash
bash courses/course1_finetuning/chapter4_dpo/class1_dpo_smollm2/run.sh
```

Smoke mode by default (~512 train pairs, ~50 steps, ~3–5 min on a single GPU). The 2× model memory means SmolLM2-135M needs ~1 GB of GPU memory; bigger backbones (360M+) want LoRA + ref-from-base.

### ⚠ Known environment requirement

**TRL ≥ 1.0 requires PyTorch ≥ 2.5** (uses `torch.distributed.fsdp.FSDPModule`, introduced in 2.5). The repo's current `env/slm-gpu.yml` pins `pytorch=2.4.*`, which means `from trl import DPOTrainer` fails with:

```
cannot import name 'FSDPModule' from 'torch.distributed.fsdp'
```

`SFTTrainer` (used by Course 1 ch1c2 / ch2 / ch3) does NOT trigger this import path and works fine. To unblock this DPO class, either:

1. **Upgrade torch in the env** (recommended, cleanest):
   ```bash
   conda install -n slm-gpu -c pytorch -c nvidia pytorch=2.6.* pytorch-cuda=12.4 -y
   ```
2. **Or downgrade trl** to a torch-2.4-compatible release:
   ```bash
   /home/Aboya_25R9803/anaconda3/envs/slm-gpu/bin/pip install 'trl==0.16.*'
   ```

After either change, re-run the smoke. The DPO code in this class is unchanged.

## How to verify

`results/full/<backbone>/course1_finetuning/chapter4_dpo_class1_dpo_smollm2/orca_dpo/dpo-b<BETA>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `train_loss_final` | 0 | 0.7 | Final DPO loss (random = 0.69 = log 2; below means signal exists) |
| `eval_loss` | 0 | 0.8 | Held-out DPO loss |
| `rewards_margin` | 0 | 100 | Mean (chosen − rejected) implicit reward — positive means policy prefers chosen |
| `rewards_accuracy` | 0.50 | 1.0 | Fraction of pairs where chosen reward > rejected — sanity that learning happened |
| `loss_decreased` | 1 | 1 | Final train loss < initial |

If `rewards_accuracy` is near 0.5 (random), the model didn't learn the preference — usually `lr` too low, `beta` too high, or `max_length` truncating the discriminative part of the completions.

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
