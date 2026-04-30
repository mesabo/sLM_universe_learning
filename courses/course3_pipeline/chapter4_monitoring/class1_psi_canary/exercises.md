# Exercises — Course 3 · ch 4 · class 1 (PSI + canary)

## 1. Warm-up — sweep the OOD ramp

Edit `live.shift_schedule` in `configs/default.yaml` to a steeper ramp (e.g. `[0.0, 0.5, 1.0]`) or a flatter ramp (`[0.1, 0.2, 0.3]`). Re-run. Does `psi_max` track the OOD fraction at the peak tick? When does the schedule become too flat for the alarm to fire?

## 2. Apply — refresh the canary periodically

The current canary is fixed at startup. Modify `train.py`: every `K` ticks, replace the oldest 25 % of the canary with newly-labeled data sampled from the live distribution. Re-run. Does `accuracy_psi_correlation` get tighter (more negative) when the canary stays representative?

## 3. Stretch — MMD as an alternative to PSI

Implement `shared.drift.mmd(baseline_emb, live_emb, sigma)` (RBF kernel, biased estimator). Run alongside PSI on the same ticks. Does MMD pick up drift earlier or later than PSI? At what computational cost?
