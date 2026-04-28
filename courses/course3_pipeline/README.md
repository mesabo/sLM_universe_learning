# Course 3 — Pipeline (deployment, active learning, monitoring)

After Course 1 (fine-tuning) and Course 2 (continual learning), Course 3 is about **what happens after the model leaves the training loop**: serving it efficiently, deciding what data to label next, swapping in updated checkpoints, and noticing when something silently regresses.

| Chapter | Topic | Why |
|---|---|---|
| `chapter1_serving/` | Inference fundamentals: single vs batched, prefill vs decode, tokens/sec | Before you reach for vLLM, know what you're optimizing |
| `chapter2_active_learning/` *(upcoming)* | Uncertainty-sampling loop on a tiny label budget | Stop labeling random data; spend each label on what the model can't predict |
| `chapter3_auto_update/` *(upcoming)* | Drift-triggered retrain + blue/green checkpoint swap | Production deployments are measured in checkpoints/week, not weights/training-step |
| `chapter4_monitoring/` *(upcoming)* | Embedding drift (PSI), canary set quality, cost / latency dashboards | If you can't see it, you can't fix it |
| `chapter5_mini_deployment/` *(upcoming)* | End-to-end mini deployment tying ch1-4 together | The capstone |

Every chapter declares a metric band; results land at `results/full/<backbone>/course3_pipeline/<class>/<task>/<method>.json`.
