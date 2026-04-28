# Exercises — Course 2 · ch1 · class 1

## 1. Warm-up — read the matrix

After `bash run.sh`, open the result JSON's `extras.history_matrix` field. It's a 3×2 grid:

```
            task0     task1
stage0    R[0][0]   R[0][1]    (before any training)
stage1    R[1][0]   R[1][1]    (after task A)
stage2    R[2][0]   R[2][1]    (after task B)
```

Verify by hand that:
- `BWT == R[2][0] - R[1][0]` (since T-1 = 1)
- `acc_A_after_B == R[2][0]`
- `avg_accuracy == (R[2][0] + R[2][1]) / 2`

## 2. Apply — closer-domain Task B

Edit `configs/default.yaml` to make Task B `sst2` (binary sentiment) instead of TREC, with `label_remap: {0: 0, 1: 1}` and rows whose mapped label is null dropped (here none are dropped, but only labels 0 and 1 of the head are used). Re-run smoke. Is BWT smaller (less forgetting) when the two tasks share more domain? Why?

## 3. Stretch — freeze the backbone

Run with `freeze_backbone: true`. This makes only the classification head trainable. Measure BWT. Compare to the full-FT baseline. Then explain in `NOTES.md`:

- Why does freezing reduce BWT?
- What does it cost you on `acc_A_after_A` and `acc_B_after_B`?
- How does this relate to chapter 4 (parameter isolation via LoRA-per-task)?
