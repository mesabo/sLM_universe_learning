# Exercises — Course 2 · ch 5 · class 1 (synthesis)

## 1. Warm-up — read your own table

Run `bash run.sh` after running ch1–4. Look at the printed table. For your seed, who wins on:
- Lowest `|BWT|`?
- Highest `acc_A_after_B`?
- Highest `avg_accuracy`?
- Highest `acc_B_after_B`?

Are these the same row? If not, what does the disagreement tell you about the trade-off?

## 2. Apply — fan across backbones

Run all four cross-backbone grids:

```bash
for g in baseline replay ewc isolation; do
  bash shared/sharded_run.sh courses/course2_continual/grids/${g}_cross_backbone.yaml
done
bash courses/course2_continual/chapter5_recipes/class1_synthesis/run.sh
```

The matrix is now ~12 cells (3 backbones × 4 methods). Does the ranking of methods stay consistent across backbones, or does (e.g.) BGE-small respond differently to EWC than MiniLM does?

## 3. Stretch — add seed sweeps and error bars

Modify the grid YAMLs to include `seed: [42, 1337, 7]`. The result JSON paths currently collide on (backbone, method) regardless of seed — fix that by adding `--method ewc-l100000-s${seed}` style suffixes (or generalize `result_path` in `shared.paths`). Re-aggregate with mean ± std per (backbone, method).
