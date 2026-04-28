# Exercises — Course 2 · ch2 · class 1

## 1. Warm-up — reproduce ch1 with this code

Override `replay.ratio=0` and re-run smoke. You should reproduce chapter 1's matrix exactly (same seed, same model, same data caps). Verify by diffing the two `extras.history_matrix` JSON fields. If they differ, what's the (likely tiny) source of nondeterminism?

## 2. Apply — sweep the ratio

Run `bash shared/sharded_run.sh courses/course2_continual/grids/replay_ratio_sweep.yaml`. This kicks off four runs on different GPUs (ratios 0.0, 0.1, 0.25, 0.5). Then:

```python
from shared.eval_harness import aggregate
df = aggregate("results/full/**/chapter2_replay_class1_simple_rehearsal/**/*.json")
print(df[["method", "metric.acc_A_after_B", "metric.BWT", "metric.acc_B_after_B"]])
```

Plot or eyeball: is BWT monotone in `ratio`? At what ratio does `acc_B_after_B` start to suffer noticeably?

## 3. Stretch — class-balanced replay

Currently we sample replay rows uniformly from Task A. Modify `_build_replay_train` to draw an equal count per class label (group-by + sample-per-group). Re-run with the same total `int(ratio * |B|)` budget. Does class-balanced replay improve BWT, or does it just slightly improve fairness across labels?
