# Notes — Course 2 · ch 2 · class 1 (simple rehearsal)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`. Free-form is fine; the headings below are starting points.

## Run log

| Run | Config / overrides | BWT | acc_A_after_B | Notes |
|---|---|---|---|---|
| baseline | `replay.ratio=0.0` |  |  |  |
| default | `replay.ratio=0.25` |  |  |  |
|  |  |  |  |  |

## Exercises

### 1. Warm-up — reproduce ch1 with this code

(What did you observe when you set `replay.ratio=0`? Did the matrix match ch1 exactly?)

### 2. Apply — sweep the ratio

(Paste the table you produced from the launcher sweep. Where does BWT plateau? Where does `acc_B_after_B` start to suffer?)

### 3. Stretch — class-balanced replay

(Describe your modification to `_build_replay_train`. Did class-balanced replay improve BWT at the same memory budget?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
