# Course 3 · Chapter 3 · Class 1 — Auto-update: drift-triggered retrain + blue/green swap

> Goal: simulate a deployed AG News classifier that retrains itself when canary accuracy drops, then atomically swaps to the new model only if it passes a quality gate. Demonstrate registry layout, promotion logic, rollback safety — the things that make deploying small models safe to do twice a week instead of once a quarter.

---

## Psycho — the mental model

> **One-line takeaway:** "deploy" is not "save the model and pray". It's a **state machine** with three states (production / candidate / archived) and three operations (train candidate / evaluate against gate / atomically swap or reject).

Most students think model deployment ends with `torch.save`. Real production systems have a much richer lifecycle: a *production* model serves predictions; a *candidate* model is trained on fresher data; the candidate must clear a **quality gate** before the system swaps it in. If the gate fails, the candidate is archived (kept for forensics) and production stays put. If the new production turns out worse later, you **roll back** by flipping a pointer.

The whole thing turns on one tiny file: `production.json`, which says "version N is current". Atomically rewriting that file is the swap. Anything else (a fancier registry, a model server) is a wrapper around this single primitive.

This class makes the lifecycle concrete with a small simulation: 4–6 cycles, synthetic drift via mixing in `dair-ai/emotion` rows, a canary set, and a margin-based gate. Watch promotions land, candidates get rejected, and the production version climb (or stay).

**Common confusion to head off:** "Why not always promote a freshly retrained model?" Because retraining doesn't always produce a *better* model — random seed variation, drifted-but-not-shifted data, or a bug in the training loop can produce a worse one. The gate is the immune system that catches those cases.

## Academic — the lifecycle and the gate

Let `M_t` be the production model at time `t` and `M_t'` be its candidate (same architecture, retrained on fresh data + existing labels). The promotion gate is:

$$
\mathrm{promote}(M_t') \iff \mathrm{acc}_{\mathrm{canary}}(M_t') \;\geq\; \mathrm{acc}_{\mathrm{canary}}(M_t) \;-\; \delta
$$

where `δ` is the **acceptance margin** — how much regression we tolerate (often a few percentage points; tighter for safety-critical, looser for fast iteration).

Two complementary trigger conditions for retraining:

- **Quality trigger**: `acc_canary(M_t) < threshold` — the production model has degraded.
- **Time trigger** (not used here, common in real systems): "retrain every N hours regardless".

The **canary set** is a labeled, fixed slice of the data distribution that's never used for training. Its accuracy is your single most important production health metric — if it falls, something is wrong even if PSI (chapter 4) hasn't fired yet.

References:
- [Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems) — why simple retraining is not enough
- [Breck et al., *The ML Test Score* (Big Data 2017)](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) — the deployment-readiness checklist
- The `production.json` atomic-swap pattern is a thin re-implementation of what every model registry (MLflow, Vertex, SageMaker) does under the hood.

## Engineering — what the code does

[`train.py`](./train.py) runs a multi-cycle simulation against `shared.registry`:

1. **Bootstrap** — train an initial classifier on `bootstrap.train_size` rows of AG News. Register as `v1` via `shared.registry.register_version`. Promote `v1` to production. Compute `canary_accuracy_at_birth` on the held-out canary set.
2. **For each cycle** in `cycles.n`:
   - Construct the live training pool by mixing `cycles.fresh_size` AG-News rows with `cycles.drift_emotion_ratio * cycles.fresh_size` Emotion rows (label-mapped to `0..3`). The Emotion fraction simulates gradual drift.
   - Compute `production_canary_acc` on the canary set.
   - **Trigger**: if `production_canary_acc < gate.degradation_threshold`, train a candidate on (existing fresh pool + previously labeled rows). Register as `v{n+1}`.
   - **Promotion gate**: candidate is promoted iff `candidate_canary_acc >= production_canary_acc - gate.acceptance_margin`. Otherwise rejected.
   - Append a structured decision row (cycle / trigger / candidate_acc / decision / reason) to `decisions.jsonl`.
3. Persist a result JSON via `shared.eval_harness.run_eval` summarizing the timeline.

### Gotchas
- **`shared/registry.py` writes to `checkpoints/<course>/<class>/<run_id>/`** — gitignored, so versions don't pollute git history. To see what was written, `tree checkpoints/course3_pipeline/...`.
- The promotion gate uses **`acceptance_margin`, not `improvement_threshold`** — we tolerate a small regression because retrain noise can dominate at small data scales. Real systems often require strict improvement on at least one metric.
- The synthetic drift here is *very* coarse (just mixing in a different dataset). Real drift detection deserves a proper ch4 (the next class) — this chapter focuses on the *response* to drift, not the *detection*.
- Smoke runs use `pool_size=1024`, `cycles.n=4` — tight but enough to see at least one promotion in expectation.

## Research — open questions / extensions

- Replace the **margin-based** gate with a **t-test** gate: candidate must beat production at p < 0.05 over a paired bootstrap of canary accuracy. Does the simulation produce fewer (more careful) promotions?
- Add **rollback on degradation**: after a promotion, if next-cycle canary accuracy drops by `> 2 * acceptance_margin`, automatically `shared.registry.rollback` to the previous version. Real systems have this as a safety net.
- The current canary is fixed at bootstrap time. Real production canaries are **periodically refreshed** (otherwise they go stale). Add `cycles.canary_refresh_period` and measure how it changes promotion behavior.
- For decoder models (instead of encoder classifiers), the gate metric isn't accuracy — it's perplexity, win-rate on a side-by-side eval, or LLM-as-judge. Sketch the gate function for that case.

---

## How to run

```bash
bash courses/course3_pipeline/chapter3_auto_update/class1_blue_green/run.sh
```

Smoke mode by default — 4 cycles on a 1024-row pool with 32-row canary, runs in ~1–2 min on a single GPU.

## How to verify

`results/full/<backbone>/course3_pipeline/chapter3_auto_update_class1_blue_green/auto_update/blue-green-c<CYCLES>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `n_cycles_completed` | 1 | 100 | Sanity that the loop ran |
| `n_drift_triggers` | 1 | 100 | The simulation actually exercised the trigger at least once |
| `n_promotions` | 0 | 100 | Number of times `production.json` flipped |
| `n_rejections` | 0 | 100 | Candidates that failed the gate |
| `final_production_version` | 1 | 100 | At least v1 always exists |
| `final_canary_accuracy` | 0.40 | 1.0 | The promotion gate must keep the production model usable |

The `extras.promotion_decisions[]` field has the full per-cycle timeline (production_canary_acc, candidate_canary_acc, decision, reason) so you can audit whether the gate behaved correctly.

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
