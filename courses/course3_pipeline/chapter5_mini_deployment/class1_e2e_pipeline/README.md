# Course 3 · Chapter 5 · Class 1 — End-to-end mini-deployment (capstone)

> Goal: combine ch1 (serving) + ch2 (active learning) + ch3 (auto-update) + ch4 (monitoring) into one runnable pipeline simulation. After this class, the student has seen what a "production sLM" lifecycle looks like in code — small simulation, real shape.

---

## Psycho — the mental model

> **One-line takeaway:** real production is *concurrent loops*. A serving loop, a monitoring loop, an active-learning loop, and an auto-update loop all run alongside each other, sharing the same model and the same data feed.

The previous four chapters each focused on one loop in isolation. This capstone runs all four at once on the same shared state, so you can see how they interact:

- **Serving** turns inputs into predictions and emits per-request **latency** and **uncertainty**.
- **Monitoring** runs every few ticks, computes **PSI** vs the embedding baseline, and tracks **live accuracy** on labeled traffic.
- **Active learning** queues high-uncertainty rows for labeling; once the queue fills, the labels become available training data.
- **Auto-update** triggers a candidate retrain when *any* of {PSI alarm, live accuracy below threshold, queue committed} fires; the candidate is promoted iff it clears the gate.

That's a closed loop: data → predictions → drift signals → labels → new model → predictions. Production teams call this the "ML data flywheel." This class makes it concrete in ~250 lines.

**Common confusion to head off:** "Why simulate when I could just deploy?" Because the failure modes only show up when components interact. PSI without an auto-update trigger is just a chart. Auto-update without monitoring is taste. Active learning without an update mechanism is wasted labels. The capstone exists because the system-level lessons are different from the component-level ones.

## Academic — the integrated trigger logic

Let `t` be the current tick. The auto-update gate fires when:

$$
\mathrm{retrain}(t) \iff
  \mathrm{PSI}(t) > \tau_{\mathrm{psi}}
  \;\lor\; \mathrm{acc}_{\mathrm{live}}(t) < \tau_{\mathrm{acc}}
  \;\lor\; |Q_{\mathrm{AL}}(t)| \geq B_{\mathrm{AL}}
$$

where `Q_AL` is the active-learning queue (high-uncertainty rows accumulated since last commit) and `B_AL` is the label-budget threshold for committing.

When `retrain(t)` is true: train a candidate on (cumulative training set ∪ committed AL labels), evaluate on the canary, promote if `candidate_canary_acc >= production_canary_acc - δ` (same gate as ch3).

References:
- [Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015)](https://papers.nips.cc/paper/5656) — the canonical "production ML is a system" paper
- The four sub-system primitives reuse the helpers from this course's prior chapters (ch1 serving, ch2 active learning, ch3 `shared/registry.py`, ch4 `shared/drift.py`).

## Engineering — what the code does

[`train.py`](./train.py) orchestrates `pipeline.n_ticks` ticks. Each tick:

1. **Serve**: pull `pipeline.live_batch_size` rows from the live stream (AG News + ramping Emotion). Predict with the production model. Measure encode latency. Compute per-row entropy.
2. **Active learning**: rows with entropy > `active.uncertainty_threshold` go into the queue. When `len(queue) >= active.label_budget`, "commit" — pull the queued rows' true labels and append to the cumulative training set.
3. **Monitor**: every `monitoring.tick_period` ticks, compute PSI vs the embedding baseline (via `shared.drift`) and live accuracy on the labeled live batch.
4. **Auto-update**: if PSI > `monitoring.alarm_threshold` OR live_acc < `gate.degradation_threshold` OR a label commit just happened, train candidate via `shared.registry`, evaluate on canary, promote/reject.

Final result JSON aggregates the four sub-system counts plus a unified `extras.tick_log[]` with per-tick (serve_count, latency, entropy, queue_size, psi?, live_acc?, decision?).

### Reuses, no new shared modules

- `shared.backbones.load_backbone` — encoder for monitoring, classifier for serving
- `shared.training.{classification_metrics, make_output_dir}` — eval + checkpoint paths
- `shared.registry.*` — model versioning + production pointer (from ch3)
- `shared.drift.{fit_projection, histogram_along_projection, psi}` — monitoring (from ch4)
- `shared.eval_harness.run_eval` — final result JSON

### Gotchas
- **The simulation is intentionally tiny** (8 ticks × 64-row batches). Real systems run for days; here we just want enough ticks for at least one of {drift alarm, AL commit, promotion} to fire.
- **The active-learning queue and the auto-update gate share a single trigger path** — if a commit and a drift alarm coincide on the same tick, you get one retrain, not two.
- **The promotion gate reuses ch3's logic verbatim** (margin-based). No need to re-tune; if it worked there, it works here.

## Research — open questions / extensions

- The four loops here are **synchronous and tick-based**. Real systems are **asynchronous** (serving is a hot path; monitoring/AL/update are background). Prototype an async version using `asyncio` and measure throughput vs the sync baseline.
- Add a **rollback** rule: if the post-promotion live_accuracy drops by more than `2 * δ` over the next K ticks, automatically `shared.registry.rollback`. Real systems need this safety net.
- Replace random AL with **margin sampling** (Course 3 ch2 exercise 2) and measure how AL-driven promotions change.
- Add a **cost knob**: each labeled row costs $X, each retrain costs $Y. Plot final live_accuracy vs total $ spent across the four loops.

---

## How to run

```bash
bash courses/course3_pipeline/chapter5_mini_deployment/class1_e2e_pipeline/run.sh
```

Smoke mode by default — 8 ticks, ~2–3 min on a single GPU.

## How to verify

`results/full/<backbone>/course3_pipeline/chapter5_mini_deployment_class1_e2e_pipeline/e2e/pipeline-t<TICKS>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `n_predictions_served` | 64 | 100000 | Sanity that the serving loop ran |
| `n_drift_alarms` | 0 | 100 | How many ticks crossed PSI > alarm_threshold |
| `n_active_learning_commits` | 0 | 100 | How many times the AL queue filled and committed |
| `n_promotions` | 0 | 100 | How many times `production.json` flipped |
| `final_production_version` | 1 | 100 | At least the bootstrap exists |
| `final_live_accuracy` | 0.20 | 1.0 | The last tick's live accuracy |
| `mean_latency_ms` | 0.0 | 600000.0 | Permissive — operational |
| `at_least_one_loop_fired` | 1 | 1 | At least one of {alarm, commit, promotion} happened (the simulation actually exercised the system) |

`extras.tick_log[]` carries the unified per-tick view so you can see which loops fired when.

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections present.
- [ ] `train.py`/`eval.py` contain no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises.
- [ ] At least one smoke run completed end-to-end and the metric band passes.
- [ ] Linked from the parent course `README.md` table.
