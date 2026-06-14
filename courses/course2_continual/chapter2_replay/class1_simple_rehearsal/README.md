# Course 2 · Chapter 2 · Class 1 — Experience Replay (Simple Rehearsal)

> Goal: reduce catastrophic forgetting by keeping a scrapbook of past examples and mixing them back in during new-task training. Observe that BWT is much closer to zero than in the baseline (ch1), without any architectural change.

---

## Psycho — the mental model

> **One-line takeaway:** experience replay = keeping a scrapbook of past homework while studying new material.

When a student re-reads a page of yesterday's chemistry notes while working through today's history chapter, the chemistry knowledge stays fresh. The model does the same thing: a compact replay buffer stores past examples, and at every training step for the new task the optimizer sees both old and new data. Because the gradient now pulls in two directions simultaneously, the classifier never fully abandons what it learned before.

In production, the replay buffer is a bounded queue sampling from your data lake. You mix replay data with the new batch at training time. This is what ML teams at Mercari and LINE actually deploy when they retrain classifiers on a monthly data stream. The buffer capacity is a budget knob: small buffer, cheap memory, slightly higher BWT; large buffer, costlier memory, near-zero BWT. Reservoir sampling (Vitter 1985) ensures every historical example has an equal shot at a buffer slot regardless of when it arrived.

A common misconception to pre-empt: replay is not fine-tuning on the old dataset. The old data is long gone. The buffer is a lossy, bounded approximation, and its statistical diversity comes from the sampling guarantee, not from keeping everything. That distinction matters for GDPR-regulated environments where the original data may be deleted after training.

## Academic — theory and math

Let $C$ be the buffer capacity and $n$ the total number of examples seen so far. Reservoir sampling guarantees that each example occupies a buffer slot with probability $\min(1, C/n)$. After seeing $n > C$ examples, the buffer holds a uniformly random subset of size $C$, making it an unbiased estimator of the empirical data distribution up to that point.

The backward transfer metric (Lopez-Paz and Ranzato, NeurIPS 2017) measures how much training on later tasks degrades earlier-task accuracy:

$$\mathrm{BWT} = \frac{1}{T-1}\sum_{i=0}^{T-2} \bigl(R_{T,i} - R_{i+1,i}\bigr)$$

where $R_{k,i}$ is test accuracy on task $i$ after finishing stage $k$. With pure sequential training (ch1), BWT is sharply negative. With replay at mixing ratio $\rho$, the effective fraction of old data in each batch is $\rho / (1 + \rho)$, which acts as an L2-like anchor on parameters that mattered for earlier tasks, pushing BWT toward zero.

Relevant references:
- [Lopez-Paz and Ranzato — *Gradient Episodic Memory for Continual Learning* (NeurIPS 2017)](https://arxiv.org/abs/1706.08840) — BWT/FWT definitions
- [Rebuffi et al. — *iCaRL: Incremental Classifier and Representation Learning* (CVPR 2017)](https://arxiv.org/abs/1611.07725) — structured exemplar replay
- [Prabhu et al. — *GDumb: A Simple Approach that Questions Our Progress in Continual Learning* (ECCV 2020)](https://arxiv.org/abs/2011.12216) — naively greedy replay as a strong baseline

## Engineering — code walk-through

`train.py` implements the following:

1. `ReplayBuffer(capacity)` — a reservoir-sampling buffer. `add(examples)` applies Vitter's algorithm: each incoming example replaces a random existing slot with probability $C/n$. `sample(n)` returns up to $n$ randomly drawn examples without replacement. `fill_rate()` returns `len(buffer) / capacity`.
2. Each task is loaded via `load_dataset` and capped at `n_samples` / `eval_n` rows. Labels are kept as-is; the shared classification head uses `max(n_labels)` outputs across all tasks.
3. For **task 0**, train normally then add all training examples to the buffer.
4. For **task 1+**, compute `n_replay = n_current * mixing_ratio / (1 - mixing_ratio)`, sample from the buffer, concatenate, shuffle, and train the mixed set. Then add task $k$'s examples to the buffer (reservoir sampling ensures capacity is never exceeded).
5. After every task, evaluate **all** tasks to build the accuracy matrix.
6. BWT and `buffer_fill_rate` are written to the result JSON via `run_eval`.

Config keys of note:

| Key | Default (smoke) | Effect |
|---|---|---|
| `replay.buffer_capacity` | 50 | Max examples kept per class stream |
| `replay.mixing_ratio` | 0.3 | Fraction of batch from replay |
| `training.save_strategy` | epoch | Enables checkpoint restore |
| `training.max_epochs` | 1 | Epochs per task (smoke) |

Memory cost estimate: `buffer_capacity * avg_seq_len * 4 bytes`. At capacity 500, max_length 128, fp32 tokens, that is roughly 256 KB — negligible.

## Research — open questions

Coreset selection (Welling 2009; Har-Peled and Roth 2014) replaces random reservoir sampling with a greedy cover algorithm: choose the $C$ examples that maximize diversity in embedding space. This gives lower reconstruction error for a fixed $C$ and typically a 5-10 pp BWT improvement at the same budget. Implement it by maintaining embeddings for all seen examples and running a greedy $k$-center selection when the buffer overflows.

DER (Buzzega et al., NeurIPS 2020) and DER++ store not just examples but also their logits at training time. The replay loss then matches current predictions to stored logits, adding a soft distillation signal on top of CE. This is compatible with the reservoir buffer and improves BWT by another 3-5 pp in typical benchmarks.

At production scale, the replay buffer can be modeled as a streaming Kafka consumer queue: each training run consumes a partition, and the buffer is a stateful operator that applies reservoir sampling over the stream. Kafka's log compaction policy provides the equivalent of a hard capacity cap without explicit eviction logic.

---

## How to run

```bash
bash courses/course2_continual/chapter2_replay/class1_simple_rehearsal/run.sh
```

Uses `configs/smoke.yaml` by default (64 samples per task, 1 epoch, buffer capacity 50). Pass a different config as the first argument:

```bash
bash run.sh configs/default.yaml
```

## How to verify

Expected metric band (smoke config):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `bwt` | -1.0 | 1.0 | Backward transfer; negative = forgetting, near-zero = retention |
| `final_acc` | 0.0 | 1.0 | Accuracy on the last task after all training |
| `buffer_fill_rate` | 0.0 | 1.0 | Fraction of buffer capacity occupied |

The smoke band is permissive because 64 samples and 1 epoch are too few for reliable convergence. Run `default.yaml` and expect `bwt` in `[-0.15, 0.05]` and `buffer_fill_rate` above 0.9.

## Instructor checklist

- [ ] All four mode sections are present and at least two paragraphs each.
- [ ] Every reference link points at an official source.
- [ ] `train.py` and `eval.py` contain no numeric literal other than `0` / `1` (all values in YAML).
- [ ] `configs/default.yaml` declares `expected_band` for every metric written by `run_eval`.
- [ ] `run.sh` sets `HF_HOME` and is `chmod +x`.
- [ ] At least one smoke-mode run completed end-to-end and the metric band passed.
- [ ] Linked from the parent course `README.md` table.
