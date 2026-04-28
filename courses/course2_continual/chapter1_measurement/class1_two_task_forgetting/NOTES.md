# Notes — Course 2 · ch 1 · class 1 (measuring catastrophic forgetting)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | Backbone | freeze_backbone | acc_A_after_A | acc_A_after_B | BWT | Notes |
|---|---|---|---|---|---|---|
| default | sentence-transformers/all-MiniLM-L6-v2 | false |  |  |  |  |
|  |  |  |  |  |  |  |

## Exercises

### 1. Warm-up — read the matrix

(Verify by hand: `BWT == R[2][0] - R[1][0]`, `acc_A_after_B == R[2][0]`, `avg_accuracy == (R[2][0] + R[2][1]) / 2`.)

### 2. Apply — closer-domain Task B

(Did BWT shrink when Task B was closer in domain to Task A? By how much, and why?)

### 3. Stretch — freeze the backbone

(With `freeze_backbone=true`: did BWT improve? What did `acc_A_after_A` and `acc_B_after_B` lose? How does this relate to chapter 4 (parameter isolation)?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
