# Exercises — Course 2 · ch 3 · class 1 (EWC)

## 1. Warm-up — sweep λ

Override at the CLI:

```bash
for lam in 0 100 1000 10000; do
  bash run.sh --config configs/default.yaml ewc.lambda=$lam
done
```

Tabulate `acc_A_after_B`, `acc_B_after_B`, and `BWT` per λ. Where's the knee?

## 2. Apply — compare to ch1 + ch2 at the same seed

Run all three (ch1 baseline, ch2 replay, ch3 EWC) with `seed=42`. Build the comparison table:

| Method | acc_A_after_B | acc_B_after_B | BWT | Notes |
|---|---|---|---|---|
| ch1 baseline | 0.46 | 0.59 | -0.34 | catastrophic |
| ch2 replay r=0.25 | 0.82 | 0.59 | +0.02 | needs Task A data |
| ch3 EWC λ=1000 | ? | ? | ? | no data needed |

Which technique is on the Pareto frontier of (BWT, acc_B_after_B)?

## 3. Stretch — replay + EWC together

Modify `train.py` to also mix in `replay.ratio` of Task A samples (port the helper from ch2). Does combining replay + EWC beat either alone? At what budget do you hit diminishing returns?
