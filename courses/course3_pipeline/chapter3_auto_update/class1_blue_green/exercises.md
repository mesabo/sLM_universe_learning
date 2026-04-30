# Exercises — Course 3 · ch 3 · class 1 (auto-update)

## 1. Warm-up — sweep the acceptance margin

```bash
for m in 0.0 0.05 0.10 0.20; do
  bash run.sh --config configs/default.yaml gate.acceptance_margin=$m
done
```

Tighter margin (`0.0`) accepts only candidates that match or beat production. Looser margin (`0.20`) tolerates large regressions. How does `n_promotions` change? At what margin does `final_canary_accuracy` drop noticeably?

## 2. Apply — add an explicit rollback rule

Modify `train.py`: at the start of each cycle, if `production_canary_acc` dropped by more than `2 * acceptance_margin` since the previous cycle, call `shared.registry.rollback(course, klass, run_id, previous_version)` *before* doing anything else. Re-run with high-drift settings (e.g. `cycles.drift_emotion_ratio=0.8`). Did the rollback ever fire?

## 3. Stretch — paired-bootstrap promotion gate

Replace the margin-based gate with a paired-bootstrap test: resample the canary set 1000 times, compute the fraction of resamples where candidate beats production. Promote only if that fraction is ≥ 0.95. Implement in `_decide_promotion(...)`. Does the more conservative gate produce fewer (better-justified) promotions?
